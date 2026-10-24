"""Streaming Generator - Line-by-line code generation with explanations."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional, List
from datetime import datetime

from code.generator.code_generator import CodeGenerator
from code.ir.project_ir import ProjectIR
from brain.explanation_events import (
    GenerationEvent, GenerationEventType, ExplanationEvent, ExplanationType
)
from brain.reasoning_events import ReasoningEventCollector
from brain.think_engine import ReasoningState
from brain.explanation_engine import ExplanationEngine


@dataclass
class StreamingGenerator:
    """Generates code line-by-line with real-time explanations by intercepting generated code."""
    
    base_generator: CodeGenerator
    explanation_engine: ExplanationEngine
    event_collector: ReasoningEventCollector

    def generate_streaming(self, 
                           project_ir: ProjectIR,
                           reasoning_state: ReasoningState,
                           language: str) -> List[dict]:
        """Generate code with line-by-line explanations (synchronous for simplicity)."""
        
        # First generate the full project to get all code
        from code.generator.project_generator import ProjectGenerator
        project_generator = ProjectGenerator()
        gen_result = project_generator.generate_project(project_ir)
        
        events = []
        current_file = None
        line_number = 0
        
        # START event
        events.append({
            "event_type": GenerationEventType.START.value,
            "timestamp": datetime.now().isoformat(),
            "file_path": "",
            "line_number": 0,
            "code": "",
            "explanation": self._explain_generation_start(project_ir, reasoning_state).to_dict(),
            "structure_context": {"purpose": f"Generate {project_ir.name} project"}
        })
        
        for f in gen_result.files:
            if f.language != language:
                continue
                
            current_file = f.path
            code_lines = f.content.split('\n')
            line_number = 0
            
            # FILE_START event
            events.append({
                "event_type": GenerationEventType.FILE_START.value,
                "timestamp": datetime.now().isoformat(),
                "file_path": f.path,
                "line_number": 0,
                "code": "",
                "explanation": self._explain_file_start(f, reasoning_state).to_dict(),
                "structure_context": {"file": f.path}
            })
            
            # Process each line
            in_class = False
            in_function = False
            current_class = None
            current_function = None
            
            for line in code_lines:
                line_number += 1
                
                # Skip empty lines but track line numbers
                if not line.strip():
                    continue
                
                # Detect structure boundaries
                stripped = line.strip()
                
                # Check for class definition
                if stripped.startswith("class ") or stripped.startswith("data class "):
                    in_class = True
                    # Extract class name
                    parts = stripped.split()
                    if len(parts) > 1:
                        current_class = parts[1].split("(")[0].split(":")[0]
                        events.append({
                            "event_type": GenerationEventType.CLASS_START.value,
                            "timestamp": datetime.now().isoformat(),
                            "file_path": f.path,
                            "line_number": line_number,
                            "code": "",
                            "explanation": self._explain_class_start(current_class, reasoning_state).to_dict(),
                            "structure_context": {"current_class": current_class, "file": f.path}
                        })
                        in_class = True
                
                # Check for function definition
                elif stripped.startswith("def ") or stripped.startswith("fun ") or stripped.startswith("function "):
                    in_function = True
                    parts = stripped.split()
                    if len(parts) > 1:
                        current_function = parts[1].split("(")[0]
                        events.append({
                            "event_type": GenerationEventType.FUNCTION_START.value,
                            "timestamp": datetime.now().isoformat(),
                            "file_path": f.path,
                            "line_number": line_number,
                            "code": "",
                            "explanation": self._explain_function_start(current_function, reasoning_state).to_dict(),
                            "structure_context": {"current_class": current_class, "current_function": current_function, "file": f.path}
                        })
                        in_function = True
                
                # Generate line event
                explanation = self._explain_code_line(line, line_number, f.path, 
                                                      {"current_class": current_class, 
                                                       "current_function": current_function,
                                                       "file": f.path},
                                                      reasoning_state)
                
                events.append({
                    "event_type": GenerationEventType.LINE.value,
                    "timestamp": datetime.now().isoformat(),
                    "file_path": f.path,
                    "line_number": line_number,
                    "code": line,
                    "explanation": explanation.to_dict() if explanation else None,
                    "structure_context": {
                        "current_class": current_class,
                        "current_function": current_function,
                        "file": f.path
                    }
                })
                
                # Check for structure end (dedentation)
                if in_class and line.startswith("    ") == False and current_class:
                    events.append({
                        "event_type": GenerationEventType.STRUCTURE_END.value,
                        "timestamp": datetime.now().isoformat(),
                        "file_path": f.path,
                        "line_number": line_number,
                        "code": "",
                        "explanation": self._explain_structure_end("class", current_class, reasoning_state).to_dict(),
                        "structure_context": {"completed": f"class:{current_class}"}
                    })
                    in_class = False
                    current_class = None
                
                if in_function and line.startswith("    ") == False and current_function:
                    events.append({
                        "event_type": GenerationEventType.STRUCTURE_END.value,
                        "timestamp": datetime.now().isoformat(),
                        "file_path": f.path,
                        "line_number": line_number,
                        "code": "",
                        "explanation": self._explain_structure_end("function", current_function, reasoning_state).to_dict(),
                        "structure_context": {"completed": f"function:{current_function}"}
                    })
                    in_function = False
                    current_function = None
        
        # COMPLETE event
        events.append({
            "event_type": GenerationEventType.COMPLETE.value,
            "timestamp": datetime.now().isoformat(),
            "file_path": "",
            "line_number": 0,
            "code": "",
            "explanation": self._explain_generation_complete(project_ir, reasoning_state).to_dict(),
            "structure_context": {"purpose": "complete"}
        })
        
        return events
    
    def _explain_generation_start(self, project_ir, state):
        return self.explanation_engine.explain_generation_start(project_ir, state)
    
    def _explain_generation_complete(self, project_ir, state):
        return self.explanation_engine.explain_generation_complete(project_ir, state)
    
    def _explain_file_start(self, file_ir, state):
        return self.explanation_engine.explain_file_start(file_ir, state)
    
    def _explain_class_start(self, class_name, state):
        from brain.explanation_types import ClassPurpose
        purpose = ClassPurpose(
            name=class_name,
            properties=[],
            reason=f"{class_name} 클래스를 정의하기 위해",
            alternatives=[]
        )
        korean = self.explanation_engine.korean.explain_class_creation(purpose)
        english = self.explanation_engine.english.explain_class_creation(purpose)
        from brain.explanation_events import ExplanationEvent, ExplanationType
        return ExplanationEvent(
            timestamp=datetime.now(),
            event_type="CLASS_START",
            file_path="",
            line_number=0,
            code="",
            explanation_type=ExplanationType.CLASS_START,
            korean_explanation=korean,
            english_explanation=english,
            decision_trace=None,
            structure_context={"current_class": class_name}
        )
    
    def _explain_function_start(self, func_name, state):
        from brain.explanation_types import FunctionPurpose
        purpose = FunctionPurpose(
            name=func_name,
            params=[],
            return_type="void",
            operations=[],
            reason=f"{func_name} 함수를 구현하기 위해",
            alternatives=[]
        )
        korean = self.explanation_engine.korean.explain_function_creation(purpose)
        english = self.explanation_engine.english.explain_function_creation(purpose)
        from brain.explanation_events import ExplanationEvent, ExplanationType
        return ExplanationEvent(
            timestamp=datetime.now(),
            event_type="FUNCTION_START",
            file_path="",
            line_number=0,
            code="",
            explanation_type=ExplanationType.FUNCTION_START,
            korean_explanation=korean,
            english_explanation=english,
            decision_trace=None,
            structure_context={"current_function": func_name}
        )
    
    def _explain_code_line(self, line, line_number, file_path, context, state):
        from brain.explanation_types import LinePurpose
        from brain.explanation_events import ExplanationEvent, ExplanationType
        
        # Simple line purpose detection
        stripped = line.strip()
        if stripped.startswith("return"):
            line_type = "return_result"
        elif stripped.startswith("if "):
            line_type = "conditional"
        elif stripped.startswith("for ") or stripped.startswith("while "):
            line_type = "loop"
        elif "=" in stripped and not stripped.startswith("if") and not stripped.startswith("for"):
            line_type = "assign_variable"
        elif "(" in stripped and ")" in stripped:
            line_type = "function_call"
        else:
            line_type = "statement"
        
        purpose = LinePurpose(line_type=line_type, node=None, context=context, detail=line)
        decision = None  # Could be enhanced to find relevant decision
        
        korean = self.explanation_engine.korean.compose_explanation(purpose, decision, context)
        english = self.explanation_engine.english.compose_explanation(purpose, decision, context)
        
        return ExplanationEvent(
            timestamp=datetime.now(),
            event_type="LINE",
            file_path=context.get("file", ""),
            line_number=line_number,
            code=line,
            explanation_type=ExplanationType.LINE,
            korean_explanation=korean,
            english_explanation=english,
            decision_trace=None,
            structure_context=context
        )
    
    def _explain_structure_end(self, struct_type, name, state):
        korean = self.explanation_engine.korean.explain_structure_completion(struct_type, name)
        english = self.explanation_engine.english.explain_structure_completion(struct_type, name)
        from brain.explanation_events import ExplanationEvent, ExplanationType
        return ExplanationEvent(
            timestamp=datetime.now(),
            event_type="STRUCTURE_END",
            file_path="",
            line_number=0,
            code="",
            explanation_type=ExplanationType.STRUCTURE_END,
            korean_explanation=korean,
            english_explanation=english,
            decision_trace=None,
            structure_context={"completed": f"{struct_type}:{name}"}
        )


# Backward compatibility - run_streaming_generation function
def run_streaming_generation(project_ir, reasoning_state, language, base_generator, explanation_engine, event_collector):
    """Synchronous wrapper for streaming generation."""
    streaming_gen = StreamingGenerator(base_generator, explanation_engine, event_collector)
    return streaming_gen.generate_streaming(project_ir, reasoning_state, language)