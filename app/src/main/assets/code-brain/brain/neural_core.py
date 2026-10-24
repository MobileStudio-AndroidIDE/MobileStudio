"""Neural Core Wrapper - Python bindings for libcodebrain.so

This module provides Python bindings to the C++ Neural Core library (libcodebrain.so),
which implements the Neural Language Core using:
- Sparse Encoder (fixed random projection -> SDR)
- Predictive Coding Stack (3-layer hierarchy with Hebbian learning)
- Semantic Memory (SDR storage + similarity search)

Graceful fallback: Neural Core is optional. If unavailable, RENA-BR
uses pure Python semantic/reasoning pipeline.
"""
from __future__ import annotations
import ctypes
import os
import sys
import platform
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class SDR:
    """Sparse Distributed Representation - 2048-bit binary vector"""
    words: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.words:
            self.words = [0] * 32

    @property
    def active_bits(self) -> List[int]:
        indices = []
        for i, word in enumerate(self.words):
            if word:
                for bit in range(64):
                    if word & (1 << bit):
                        indices.append(i * 64 + bit)
        return indices

    def count(self) -> int:
        return sum(bin(w).count('1') for w in self.words)

    def jaccard(self, other: 'SDR') -> float:
        inter = sum(bin(a & b).count('1') for a, b in zip(self.words, other.words))
        a = self.count()
        b = other.count()
        uni = a + b - inter
        return inter / uni if uni > 0 else 0.0

    def overlap(self, other: 'SDR') -> int:
        return sum(bin(a & b).count('1') for a, b in zip(self.words, other.words))


@dataclass
class DenseVector:
    """Dense vector for neural activations"""
    data: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> float:
        return self.data[i]

    def __setitem__(self, i: int, value: float):
        self.data[i] = value


@dataclass
class NeuralCoreStatus:
    """Runtime status of Neural Core availability"""
    available: bool = False
    backend: str = "fallback"
    library_path: Optional[str] = None
    error: Optional[str] = None
    version: Optional[str] = None
    build_info: Optional[str] = None


class _NeuralCoreUnavailable:
    """Sentinel for unavailable neural core"""
    pass


NEURAL_CORE_UNAVAILABLE = _NeuralCoreUnavailable()


def _is_android() -> bool:
    """Detect Android/Termux environment"""
    return (
        platform.system() == "Linux" and
        (
            "android" in platform.platform().lower() or
            "ANDROID_ROOT" in os.environ or
            "TERMUX_VERSION" in os.environ or
            os.path.exists("/system/bin/linker") or
            os.path.exists("/system/bin/linker64")
        )
    )


def _is_termux() -> bool:
    """Detect Termux environment"""
    return "TERMUX_VERSION" in os.environ or os.path.exists("/data/data/com.termux")


def _get_termux_lib_dir() -> Optional[Path]:
    """Get Termux library directory where we can load .so files"""
    if _is_termux():
        prefix = os.environ.get("PREFIX", "/data/data/com.termux/files/usr")
        return Path(prefix) / "lib"
    return None


def _get_possible_library_paths() -> List[str]:
    """Get possible library paths in order of preference"""
    paths = []

    # 1. Environment variable override (highest priority)
    env_path = os.environ.get("CODEBRAIN_NEURAL_CORE_PATH")
    if env_path:
        paths.append(env_path)

    # 2. Termux lib directory (Android-safe location)
    termux_lib = _get_termux_lib_dir()
    if termux_lib:
        paths.append(str(termux_lib / "libcodebrain.so"))

    # 3. Local build directory (development)
    paths.append(str(Path(__file__).parent.parent.parent / "core" / "build" / "libcodebrain.so"))
    paths.append(str(Path(__file__).parent.parent.parent / "core" / "build" / "Release" / "libcodebrain.so"))

    # 4. System library paths
    paths.append("/usr/local/lib/libcodebrain.so")
    paths.append("/usr/lib/libcodebrain.so")

    # 5. Current directory
    paths.append("./libcodebrain.so")
    paths.append("./core/build/libcodebrain.so")

    return paths


def _find_library() -> Optional[str]:
    """Find libcodebrain.so in accessible locations"""
    for path in _get_possible_library_paths():
        try:
            p = Path(path)
            if p.exists() and os.access(p, os.R_OK):
                return str(p)
        except (OSError, PermissionError):
            continue
    return None


def _try_load_library(path: str) -> Optional[ctypes.CDLL]:
    """Try to load library, return None on failure"""
    try:
        lib = ctypes.CDLL(path)
        return lib
    except (OSError, PermissionError) as e:
        return None


class NeuralCore:
    """Python wrapper for libcodebrain.so Neural Core

    If library cannot be loaded, methods raise RuntimeError.
    Use NeuralCore.is_available() or get_neural_core_status() to check.
    """

    _instance: Optional['NeuralCore'] = None
    _status: NeuralCoreStatus = NeuralCoreStatus()
    _lib: Optional[ctypes.CDLL] = None

    def __new__(cls, library_path: Optional[str] = None) -> 'NeuralCore':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, library_path: Optional[str] = None):
        if self._initialized:
            return

        self._initialized = True
        self._engine = None

        # Find and load library
        if library_path:
            self._lib = _try_load_library(library_path)
            if self._lib:
                self._status.library_path = library_path
            else:
                self._status.error = f"Failed to load specified library: {library_path}"
        else:
            found_path = _find_library()
            if found_path:
                self._lib = _try_load_library(found_path)
                if self._lib:
                    self._status.library_path = found_path
                else:
                    self._status.error = f"Found library but cannot load (namespace restriction): {found_path}"
            else:
                self._status.error = "libcodebrain.so not found in any search path"

        if self._lib is None:
            self._status.available = False
            self._status.backend = "fallback"
            return

        # Setup types and initialize
        try:
            self._setup_types()
            self._initialize()
            self._status.available = True
            self._status.backend = "native"

            # Get version info
            try:
                if hasattr(self._lib, 'codebrain_version'):
                    self._status.version = self._lib.codebrain_version().decode('utf-8')
                if hasattr(self._lib, 'codebrain_build_info'):
                    self._status.build_info = self._lib.codebrain_build_info().decode('utf-8')
            except Exception:
                pass

        except Exception as e:
            self._status.available = False
            self._status.backend = "fallback"
            self._status.error = f"Initialization failed: {e}"
            self._lib = None
            self._engine = None

    def _setup_types(self):
        class CodeBrainSDR(ctypes.Structure):
            _fields_ = [("words", ctypes.c_uint64 * 32)]

        class CodeBrainVector(ctypes.Structure):
            _fields_ = [
                ("data", ctypes.POINTER(ctypes.c_float)),
                ("size", ctypes.c_size_t),
            ]

        class CodeBrainConfig(ctypes.Structure):
            _fields_ = [
                ("seed", ctypes.c_uint64),
                ("capacity", ctypes.c_size_t),
                ("learning_rate", ctypes.c_float),
            ]

        self.CodeBrainSDR = CodeBrainSDR
        self.CodeBrainVector = CodeBrainVector
        self.CodeBrainConfig = CodeBrainConfig

        self.lib = self._lib

        self.lib.codebrain_engine_create.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.codebrain_engine_create.restype = ctypes.c_int

        self.lib.codebrain_engine_destroy.argtypes = [ctypes.c_void_p]

        self.lib.codebrain_encode_text.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)
        ]
        self.lib.codebrain_encode_text.restype = ctypes.c_int

        self.lib.codebrain_encode_text_dense.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)
        ]
        self.lib.codebrain_encode_text_dense.restype = ctypes.c_int

        self.lib.codebrain_process.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)
        ]
        self.lib.codebrain_process.restype = ctypes.c_int

        self.lib.codebrain_learn.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p
        ]
        self.lib.codebrain_learn.restype = ctypes.c_int

        self.lib.codebrain_stack_create.argtypes = [ctypes.c_uint64]
        self.lib.codebrain_stack_create.restype = ctypes.c_void_p

        self.lib.codebrain_stack_destroy.argtypes = [ctypes.c_void_p]

        self.lib.codebrain_stack_encode.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(CodeBrainVector)
        ]
        self.lib.codebrain_stack_encode.restype = ctypes.c_int

        self.lib.codebrain_stack_state.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.codebrain_stack_state.restype = ctypes.c_int

        self.lib.codebrain_stack_reset.argtypes = [ctypes.c_void_p]

        self.lib.codebrain_stack_set_context.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
        ]

        self.lib.codebrain_sdr_overlap.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t)
        ]
        self.lib.codebrain_sdr_overlap.restype = ctypes.c_int

        self.lib.codebrain_sdr_jaccard.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.codebrain_sdr_jaccard.restype = ctypes.c_int

        self.lib.codebrain_sdr_count.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)
        ]
        self.lib.codebrain_sdr_count.restype = ctypes.c_int

        self.lib.codebrain_memory_create.argtypes = [ctypes.c_size_t]
        self.lib.codebrain_memory_create.restype = ctypes.c_void_p

        self.lib.codebrain_memory_destroy.argtypes = [ctypes.c_void_p]

        self.lib.codebrain_memory_store.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p
        ]
        self.lib.codebrain_memory_store.restype = ctypes.c_int

        self.lib.codebrain_memory_retrieve.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_size_t)
        ]
        self.lib.codebrain_memory_retrieve.restype = ctypes.c_int

        self.lib.codebrain_memory_free_results.argtypes = [
            ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
        ]

        self.lib.codebrain_encoder_create.argtypes = [ctypes.c_uint64]
        self.lib.codebrain_encoder_create.restype = ctypes.c_void_p

        self.lib.codebrain_encoder_destroy.argtypes = [ctypes.c_void_p]

        self.lib.codebrain_encoder_encode_text.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)
        ]
        self.lib.codebrain_encoder_encode_text.restype = ctypes.c_int

        self.lib.codebrain_encoder_encode_tokens.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)
        ]
        self.lib.codebrain_encoder_encode_tokens.restype = ctypes.c_int

        self.lib.codebrain_version.restype = ctypes.c_char_p
        self.lib.codebrain_build_info.restype = ctypes.c_char_p

        self.lib.codebrain_vector_create.argtypes = [ctypes.c_size_t]
        self.lib.codebrain_vector_create.restype = ctypes.c_void_p
        self.lib.codebrain_vector_destroy.argtypes = [ctypes.c_void_p]

    def _initialize(self):
        config = self.CodeBrainConfig()
        config.seed = 0xDEADBEEFCAFEBABE
        config.capacity = 10000
        config.learning_rate = 0.01

        engine_ptr = ctypes.c_void_p()
        result = self.lib.codebrain_engine_create(
            ctypes.byref(ctypes.c_void_p(ctypes.addressof(config))),
            ctypes.byref(engine_ptr)
        )
        if result != 0:
            raise RuntimeError(f"Failed to create engine: {result}")
        self._engine = engine_ptr

    @classmethod
    def get_status(cls) -> NeuralCoreStatus:
        """Get current neural core status"""
        return cls._status

    @classmethod
    def is_available(cls) -> bool:
        """Check if neural core is available"""
        return cls._status.available

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing)"""
        if cls._instance and cls._instance._engine:
            try:
                cls._instance.lib.codebrain_engine_destroy(cls._instance._engine)
            except Exception:
                pass
        cls._instance = None
        cls._status = NeuralCoreStatus()
        cls._lib = None

    def encode_text(self, text: str) -> SDR:
        if not self._status.available:
            raise RuntimeError("Neural Core not available")
        sdr = self.CodeBrainSDR()
        result = self.lib.codebrain_encode_text(self._engine, text.encode('utf-8'), ctypes.byref(sdr))
        if result != 0:
            raise RuntimeError(f"Encode failed: {result}")
        return self._ctypes_sdr_to_sdr(sdr)

    def _ctypes_sdr_to_sdr(self, c_sdr) -> SDR:
        words = [c_sdr.words[i] for i in range(32)]
        return SDR(words=words)

    def encode_text_dense(self, text: str) -> DenseVector:
        if not self._status.available:
            raise RuntimeError("Neural Core not available")
        vec = self.CodeBrainVector()
        result = self.lib.codebrain_encode_text_dense(self._engine, text.encode('utf-8'), ctypes.byref(vec))
        if result != 0:
            raise RuntimeError(f"Encode dense failed: {result}")
        data = [vec.data[i] for i in range(vec.size)]
        return DenseVector(data=data)

    def process(self, text: str) -> str:
        if not self._status.available:
            raise RuntimeError("Neural Core not available")
        response_ptr = ctypes.c_char_p()
        result = self.lib.codebrain_process(self._engine, text.encode('utf-8'), ctypes.byref(response_ptr))
        if result != 0:
            raise RuntimeError(f"Process failed: {result}")
        response = response_ptr.value.decode('utf-8')
        return response

    def learn(self, input_text: str, target: str = "") -> bool:
        if not self._status.available:
            raise RuntimeError("Neural Core not available")
        result = self.lib.codebrain_learn(self._engine, input_text.encode('utf-8'), target.encode('utf-8'))
        return result == 0

    def create_stack(self, seed: int = 42):
        if not self._status.available:
            raise RuntimeError("Neural Core not available")
        stack_ptr = self.lib.codebrain_stack_create(seed)
        if not stack_ptr:
            raise RuntimeError("Failed to create stack")
        return Stack(self, stack_ptr)

    def destroy(self):
        if self._engine:
            self.lib.codebrain_engine_destroy(self._engine)
            self._engine = None

    def think(self, request: str, language: str = "python", context: dict = None) -> "ReasoningState":
        """Use Neural Core to understand request and return structured reasoning state."""
        if not self._status.available:
            raise RuntimeError("Neural Core not available")

        stack = self.create_stack()

        stack.encode(request)

        state = stack.state()

        from brain.think_engine import ReasoningState
        from brain.think_engine import GoalType, Entity, Operation, Constraint, ProblemModel

        reasoning_state = ReasoningState()
        reasoning_state.model.raw_request = request
        reasoning_state.model.language = language

        goal_type = self._infer_goal_type(request)
        reasoning_state.model.goal_type = goal_type

        entities = self._extract_entities(request)
        operations = self._extract_operations(request)
        constraints = self._extract_constraints(request)

        reasoning_state.model = ProblemModel(
            goal_type=goal_type,
            summary=request[:200],
            entities=[{"name": e["name"], "kind": e["kind"], "attributes": e["attributes"]}
                      for e in entities],
            operations=[{"name": o["name"], "verb": o["verb"], "category": o["category"]}
                       for o in operations],
            constraints=[{"name": c[0], "value": c[1], "kind": c[2]} for c in constraints],
            domain=entities[0]["name"].lower() if entities else "general",
            language=language,
            platform=self._detect_platform(request),
            raw_request=request,
        )

        reasoning_state.record("understand", f"Neural understanding: {goal_type.value}",
                      [f"entities={len(entities)}", f"ops={len(operations)}"], 0.9)

        return reasoning_state

    def _infer_goal_type(self, request: str) -> "GoalType":
        from brain.think_engine import GoalType
        request_lower = request.lower()

        if any(k in request_lower for k in ["project", "app", "application", "system", "program",
                                            "software", "tool", "manager", "organizer", "calculator",
                                            "todo", "website", "프로젝트", "앱", "시스템", "프로그램"]):
            return GoalType.CREATE_PROJECT
        if any(k in request_lower for k in ["class", "struct", "클래스", "객체"]):
            return GoalType.CREATE_CLASS
        if any(k in request_lower for k in ["function", "method", "func", "함수", "메소드", "메서드"]):
            return GoalType.CREATE_FUNCTION
        if any(k in request_lower for k in ["fix", "debug", "error", "bug", "오류", "고쳐", "수정"]):
            return GoalType.FIX
        if any(k in request_lower for k in ["explain", "설명", "이해"]):
            return GoalType.EXPLAIN
        if any(k in request_lower for k in ["modify", "change", "update", "add", "remove", "delete",
                                             "삭제", "추가"]):
            return GoalType.MODIFY
        if any(k in request_lower for k in ["analyze", "review", "분석"]):
            return GoalType.ANALYZE
        if any(k in request_lower for k in ["code", "implement", "generate", "create",
                                             "만들", "짜"]):
            return GoalType.CREATE_FUNCTION
        return GoalType.UNKNOWN

    def _extract_entities(self, request: str) -> List[dict]:
        entities = []
        request_lower = request.lower()

        patterns = [
            ("student", "Student", "data"), ("학생", "Student", "data"),
            ("contact", "Contact", "data"), ("연락처", "Contact", "data"),
            ("todo", "TodoItem", "data"), ("할일", "TodoItem", "data"),
            ("task", "Task", "data"), ("작업", "Task", "data"),
            ("user", "User", "data"), ("유저", "User", "data"),
            ("employee", "Employee", "data"), ("직원", "Employee", "data"),
            ("product", "Product", "data"), ("상품", "Product", "data"),
            ("order", "Order", "data"), ("주문", "Order", "data"),
            ("camera", "Camera", "device"), ("카메라", "Camera", "device"),
            ("file", "File", "resource"), ("파일", "File", "resource"),
            ("image", "Image", "resource"), ("이미지", "Image", "resource"),
            ("calculator", "Calculator", "utility"), ("계산기", "Calculator", "utility"),
            ("function", "Function", "code"), ("함수", "Function", "code"),
            ("method", "Method", "code"), ("메서드", "Method", "code"),
            ("class", "Class", "code"), ("클래스", "Class", "code"),
        ]

        seen = set()
        for kw, name, kind in patterns:
            if kw in request_lower and name not in seen:
                seen.add(name)
                entities.append({"name": name, "kind": kind, "attributes": {}})
        return entities

    def _extract_operations(self, request: str) -> List[dict]:
        operations = []
        request_lower = request.lower()

        patterns = [
            ("add", "add", "crud"), ("create", "add", "crud"), ("register", "add", "crud"),
            ("추가", "add", "crud"), ("등록", "add", "crud"), ("생성", "add", "crud"), ("만들", "add", "crud"),
            ("remove", "remove", "crud"), ("delete", "remove", "crud"),
            ("삭제", "remove", "crud"), ("제거", "remove", "crud"),
            ("update", "update", "crud"), ("modify", "update", "crud"),
            ("edit", "update", "crud"), ("수정", "update", "crud"), ("변경", "update", "crud"),
            ("search", "search", "query"), ("find", "search", "query"),
            ("lookup", "search", "query"), ("검색", "search", "query"),
            ("찾", "search", "query"), ("list", "list", "query"),
            ("show", "list", "query"), ("display", "list", "query"),
            ("목록", "list", "query"), ("보기", "list", "query"),
            ("sort", "sort", "algorithm"), ("정렬", "sort", "algorithm"),
            ("filter", "filter", "query"), ("필터", "filter", "query"),
            ("calculate", "calculate", "compute"), ("compute", "calculate", "compute"),
            ("계산", "calculate", "compute"), ("sum", "sum", "compute"),
            ("합", "sum", "compute"), ("average", "average", "compute"),
            ("평균", "average", "compute"), ("max", "max", "compute"),
            ("최대", "max", "compute"), ("min", "min", "compute"),
            ("최소", "min", "compute"), ("save", "store", "persist"),
            ("store", "store", "persist"), ("저장", "store", "persist"),
            ("load", "load", "persist"), ("불러오기", "load", "persist"),
        ]

        seen = set()
        for kw, name, cat in patterns:
            if kw in request_lower and name not in seen:
                seen.add(name)
                operations.append({"name": name, "verb": name, "category": cat})

        if not operations:
            entity_kws = ["student", "contact", "todo", "user", "employee",
                         "product", "order", "camera", "file", "image",
                         "calculator", "계산기", "학생", "연락처", "할일",
                         "사용자", "직원", "상품", "주문", "카메라", "파일", "이미지"]
            for kw in entity_kws:
                if kw in request_lower:
                    operations = [
                        {"name": "add", "verb": "add", "category": "crud"},
                        {"name": "remove", "verb": "remove", "category": "crud"},
                        {"name": "update", "verb": "update", "category": "crud"},
                        {"name": "search", "verb": "search", "category": "query"},
                        {"name": "list", "verb": "list", "category": "query"},
                    ]
                    break
        return operations

    def _extract_constraints(self, request: str) -> List[tuple]:
        constraints = []
        request_lower = request.lower()

        lang_patterns = {
            "python": "python", "파이썬": "python",
            "kotlin": "kotlin", "코틀린": "kotlin",
            "java": "java", "자바": "java",
            "javascript": "javascript", "자바스크립트": "javascript",
            "js": "javascript", "node": "javascript",
            "c++": "cpp", "cpp": "cpp",
            "c": "c",
            "go": "go", "golang": "go",
            "rust": "rust", "러스트": "rust",
            "c#": "csharp", "csharp": "csharp",
        }
        for kw, lang in lang_patterns.items():
            if kw in request_lower:
                constraints.append(("language", lang, "language"))
                break

        plat_patterns = {
            "android": "android", "안드로이드": "android",
            "mobile": "android", "모바일": "android",
            "web": "web", "웹": "web",
            "browser": "web", "브라우저": "web",
            "cli": "cli", "command line": "cli",
            "명령줄": "cli", "콘솔": "cli", "터미널": "cli",
            "backend": "backend", "서버": "backend",
            "server": "backend", "api": "backend",
        }
        for kw, plat in plat_patterns.items():
            if kw in request_lower:
                constraints.append(("platform", plat, "platform"))
                break

        if "django" in request_lower:
            constraints.append(("django", "django", "framework"))
        if "flask" in request_lower:
            constraints.append(("flask", "flask", "framework"))
        if "spring" in request_lower:
            constraints.append(("spring", "spring", "framework"))
        if "react" in request_lower:
            constraints.append(("react", "react", "framework"))

        if "fast" in request_lower or "성능" in request_lower or "빠른" in request_lower:
            constraints.append(("performance", "high", "quality"))
        if "secure" in request_lower or "보안" in request_lower:
            constraints.append(("security", "high", "quality"))

        return constraints

    def _detect_platform(self, request: str) -> str:
        request_lower = request.lower()
        if "android" in request_lower or "안드로이드" in request_lower or \
           "mobile" in request_lower or "모바일" in request_lower:
            return "android"
        if "web" in request_lower or "웹" in request_lower or \
           "browser" in request_lower:
            return "web"
        if "cli" in request_lower or "command" in request_lower or \
           "terminal" in request_lower or "명령줄" in request_lower:
            return "cli"
        if "backend" in request_lower or "server" in request_lower or \
           "api" in request_lower or "백엔드" in request_lower:
            return "backend"
        return "cli"


class Stack:
    """PC Stack wrapper"""

    def __init__(self, neural_core: NeuralCore, stack_ptr: ctypes.c_void_p):
        self.neural_core = neural_core
        self._stack_ptr = ctypes.c_void_p(stack_ptr)

    def encode(self, text: str) -> DenseVector:
        vec = self.neural_core.CodeBrainVector()
        result = self.neural_core.lib.codebrain_stack_encode(
            self._stack_ptr, text.encode('utf-8'), ctypes.byref(vec)
        )
        if result != 0:
            raise RuntimeError(f"Stack encode failed: {result}")
        data = [vec.data[i] for i in range(vec.size)]
        return DenseVector(data=data)

    def encode_sdr(self, sdr: SDR) -> DenseVector:
        raise NotImplementedError("SDR encode not yet implemented")

    def learn(self, input_text: str, target: str = "") -> bool:
        result = self.neural_core.lib.codebrain_stack_learn(
            self._stack_ptr, input_text.encode('utf-8'), target.encode('utf-8')
        )
        return result == 0

    def state(self) -> Dict[str, Any]:
        l1_act = (ctypes.c_float * 1024)()
        l2_act = (ctypes.c_float * 512)()
        l3_act = (ctypes.c_float * 256)()
        l1_err = (ctypes.c_float * 1024)()
        l2_err = (ctypes.c_float * 512)()
        l3_err = (ctypes.c_float * 256)()
        total_err = ctypes.c_float()

        result = self.neural_core.lib.codebrain_stack_state(
            self._stack_ptr, l1_act, l2_act, l3_act,
            l1_err, l2_err, l3_err,
            ctypes.byref(total_err)
        )
        if result != 0:
            raise RuntimeError(f"Stack state failed: {result}")

        return {
            "layer1_activity": list(l1_act),
            "layer2_activity": list(l2_act),
            "layer3_activity": list(l3_act),
            "layer1_error": list(l1_err),
            "layer2_error": list(l2_err),
            "layer3_error": list(l3_err),
            "total_error": total_err.value,
        }

    def reset(self):
        self.neural_core.lib.codebrain_stack_reset(self._stack_ptr)

    def set_context(self, context: DenseVector):
        arr = (ctypes.c_float * 256)(*context.data)
        self.neural_core.lib.codebrain_stack_set_context(
            self._stack_ptr, arr, 256
        )


def get_neural_core(library_path: Optional[str] = None):
    """Get neural core instance. NEVER crashes - returns NEURAL_CORE_UNAVAILABLE sentinel if unavailable."""
    if NeuralCore._instance is None:
        try:
            NeuralCore(library_path)
        except Exception:
            pass

    if NeuralCore._instance and NeuralCore._status.available:
        return NeuralCore._instance
    return NEURAL_CORE_UNAVAILABLE


def get_neural_core_status() -> NeuralCoreStatus:
    """Get explicit runtime status of Neural Core"""
    # Force initialization if not done
    if NeuralCore._instance is None:
        try:
            NeuralCore()
        except Exception:
            pass
    return NeuralCore.get_status()


def encode_text(text: str) -> SDR:
    core = get_neural_core()
    if core is NEURAL_CORE_UNAVAILABLE:
        raise RuntimeError("Neural Core not available")
    return core.encode_text(text)


def encode_text_dense(text: str) -> DenseVector:
    core = get_neural_core()
    if core is NEURAL_CORE_UNAVAILABLE:
        raise RuntimeError("Neural Core not available")
    return core.encode_text_dense(text)


def process(text: str) -> str:
    core = get_neural_core()
    if core is NEURAL_CORE_UNAVAILABLE:
        raise RuntimeError("Neural Core not available")
    return core.process(text)


def learn(input_text: str, target: str = "") -> bool:
    core = get_neural_core()
    if core is NEURAL_CORE_UNAVAILABLE:
        raise RuntimeError("Neural Core not available")
    return core.learn(input_text, target)


def create_stack(seed: int = 42) -> Stack:
    core = get_neural_core()
    if core is NEURAL_CORE_UNAVAILABLE:
        raise RuntimeError("Neural Core not available")
    return core.create_stack(seed)


if __name__ == "__main__":
    print("=== Neural Core Status ===")
    status = get_neural_core_status()
    print(f"Available: {status.available}")
    print(f"Backend: {status.backend}")
    print(f"Library Path: {status.library_path}")
    print(f"Error: {status.error}")
    print(f"Version: {status.version}")
    print(f"Build Info: {status.build_info}")

    if status.available:
        nc = NeuralCore()
        print("\n=== Neural Core Test ===")
        sdr = nc.encode_text("두 숫자를 더하는 함수를 만들어줘")
        print(f"SDR active bits: {len(sdr.active_bits())}")

        result = nc.process("두 숫자를 더하는 함수를 만들어줘")
        print(f"Process result: {result[:200]}...")

        stack = nc.create_stack(42)
        vec = stack.encode("test input")
        print(f"Stack encode size: {len(vec.data)}")

        state = stack.state()
        print(f"Stack state - total error: {state['total_error']}")

        print("Neural Core test passed!")
    else:
        print("\nNeural Core not available - using fallback mode")
        print("RENA-BR will use pure Python semantic/reasoning pipeline")