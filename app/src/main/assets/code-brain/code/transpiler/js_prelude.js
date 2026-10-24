/* RENA-BR javascript runtime prelude: python-compatible semantics.
 *
 * Invariants:
 * - Top level uses only function declarations, var and $-prefixed class
 *   names, so generated ``var len = ...`` redeclarations (python builtin
 *   shadowing) stay legal; python identifiers can never contain "$".
 * - Integers: values with |v| <= 2^53-1 are Numbers, larger exact
 *   integers are BigInts; helpers always convert results back to Number
 *   when safe so loops/indices stay fast Numbers.
 * - Attribute helpers live on the collision-proof "$" namespace.
 */

var __name__ = "__main__";
var __debug__ = true;

function pyerr(pytype, msg) {
  var e = new Error(msg);
  e.pytype = pytype;
  return e;
}

function typename(v) {
  if (v === null || v === undefined) return "NoneType";
  var t = typeof v;
  if (t === "boolean") return "bool";
  if (t === "number") return Number.isInteger(v) ? "int" : "float";
  if (t === "bigint") return "int";
  if (t === "string") return "str";
  if (t === "function") return "function";
  if (Array.isArray(v)) return v.__pytup !== undefined ? "tuple" : "list";
  if (v instanceof $Dict) return v.__pycnt !== undefined ? "Counter" : "dict";
  if (v instanceof $Set) return "set";
  if (v instanceof $Deque) return "deque";
  if (v instanceof $Range) return "range";
  if (v instanceof Map) return "dict";
  if (v instanceof Set) return "set";
  if (v && v.$name) return v.$name;
  if (t === "object") return "object";
  return t;
}

// ---------------------------------------------------------------------------
// stdin / stdout (line buffered over one shared buffer, writeSync so an
// early process.exit() never loses output)
// ---------------------------------------------------------------------------
var __py$inbuf = null;
var __py$inpos = 0;

function __py$stdinBuf() {
  if (__py$inbuf === null) {
    try {
      __py$inbuf = require("fs").readFileSync(0);
    } catch (e) {
      __py$inbuf = Buffer.alloc(0);
    }
  }
  return __py$inbuf;
}

function __py$stdinRest() {
  var b = __py$stdinBuf();
  return __py$inpos >= b.length ? "" : b.toString("utf8", __py$inpos);
}

function __py$stdinLine() {
  var b = __py$stdinBuf();
  if (__py$inpos >= b.length) return null;
  var i = b.indexOf(10, __py$inpos);
  var end = i < 0 ? b.length : i;
  var s = b.toString("utf8", __py$inpos, end);
  __py$inpos = i < 0 ? b.length : i + 1;
  if (s.length > 0 && s.charCodeAt(s.length - 1) === 13) s = s.slice(0, -1);
  return s;
}

function out(s) {
  var b = Buffer.from(s, "utf8");
  var off = 0;
  var fs = require("fs");
  while (off < b.length) off += fs.writeSync(1, b, off, b.length - off);
}

// ---------------------------------------------------------------------------
// exact integer helpers (Number <= 2^53-1, BigInt beyond)
// ---------------------------------------------------------------------------
var __py$S = 9007199254740991;          // 2^53 - 1

function __py$n(big) {
  return (big >= -9007199254740991n && big <= 9007199254740991n)
    ? Number(big) : big;
}

function __py$intish(v) {
  var t = typeof v;
  if (t === "bigint") return true;
  return t === "number" && Number.isInteger(v);
}

function __py$bi(v) {                    // -> exact BigInt or null
  if (typeof v === "bigint") return v;
  if (typeof v === "boolean") return v ? 1n : 0n;
  if (typeof v === "number" && Number.isInteger(v)) return BigInt(v);
  return null;
}

function pyadd(a, b) {
  if (typeof a === "boolean") a = a ? 1 : 0;
  if (typeof b === "boolean") b = b ? 1 : 0;
  if (typeof a === "number" && typeof b === "number") {
    if (Number.isInteger(a) && Number.isInteger(b)) {
      var r = a + b;
      if (Number.isSafeInteger(r)) return r;
      return __py$n(BigInt(a) + BigInt(b));
    }
    return a + b;
  }
  var ba = __py$bi(a), bb = __py$bi(b);
  if (ba !== null && bb !== null) return __py$n(ba + bb);
  if (typeof a === "string" && typeof b === "string") return a + b;
  if (a instanceof $Dict || b instanceof $Dict) {
    var ca = a instanceof $Dict && a.__pycnt !== undefined;
    var cb = b instanceof $Dict && b.__pycnt !== undefined;
    if (ca || cb) {                     // Counter + mapping -> Counter
      var acc = new Map();
      var i, e, k;
      if (a instanceof $Dict) for (e of a) acc.set(e[0], e[1]);
      if (b instanceof $Dict) for (e of b) {
        acc.set(e[0], acc.has(e[0]) ? pyadd(acc.get(e[0]), e[1]) : e[1]);
      }
      var res = new $Counter();
      for (e of acc) if (typeof e[1] === "number" ? e[1] > 0
          : (e[1] > 0n)) res.set(e[0], e[1]);
      return res;
    }
  }
  throw pyerr("TypeError", "unsupported operand type(s) for +: '" +
    typename(a) + "' and '" + typename(b) + "'");
}

function pysub(a, b) {
  if (typeof a === "boolean") a = a ? 1 : 0;
  if (typeof b === "boolean") b = b ? 1 : 0;
  if (typeof a === "number" && typeof b === "number") {
    if (Number.isInteger(a) && Number.isInteger(b)) {
      var r = a - b;
      if (Number.isSafeInteger(r)) return r;
      return __py$n(BigInt(a) - BigInt(b));
    }
    return a - b;
  }
  var ba = __py$bi(a), bb = __py$bi(b);
  if (ba !== null && bb !== null) return __py$n(ba - bb);
  if (a instanceof Set && b instanceof Set) {
    var d = new $Set();
    for (var x of a) if (!b.has(x)) d.add(x);
    return d;
  }
  if (a instanceof $Dict && a.__pycnt !== undefined &&
      b instanceof $Dict && b.__pycnt !== undefined) {
    var res = new $Counter();
    for (var e of a) {
      var v = b.has(e[0]) ? pysub(e[1], b.get(e[0])) : e[1];
      if (v > 0) res.set(e[0], v);
    }
    return res;
  }
  throw pyerr("TypeError", "unsupported operand type(s) for -: '" +
    typename(a) + "' and '" + typename(b) + "'");
}

function strmul(s, n) {
  if (typeof n === "bigint") n = Number(n);
  if (typeof n !== "number" || !Number.isInteger(n)) {
    throw pyerr("TypeError",
      "can't multiply sequence by non-int of type '" + typename(n) + "'");
  }
  if (n <= 0) return "";
  var r = "";
  var chunk = s;
  while (n > 0) {
    if (n & 1) r += chunk;
    n >>= 1;
    chunk += chunk;
  }
  return r;
}

function pymul(a, b) {
  if (typeof a === "boolean") a = a ? 1 : 0;
  if (typeof b === "boolean") b = b ? 1 : 0;
  if (typeof a === "number" && typeof b === "number") {
    if (Number.isInteger(a) && Number.isInteger(b)) {
      if (a === 0 || b === 0) return 0;
      if (Math.abs(a) * Math.abs(b) <= 9007199254740991) {
        var r = a * b;
        return Number.isSafeInteger(r) ? r : __py$n(BigInt(a) * BigInt(b));
      }
      return __py$n(BigInt(a) * BigInt(b));
    }
    return a * b;
  }
  var ba = __py$bi(a), bb = __py$bi(b);
  if (ba !== null && bb !== null) return __py$n(ba * bb);
  if (typeof a === "string") return strmul(a, b);
  if (typeof b === "string") return strmul(b, a);
  if (Array.isArray(a)) return repeat(a, b);
  if (Array.isArray(b)) return repeat(b, a);
  throw pyerr("TypeError", "unsupported operand type(s) for *: '" +
    typename(a) + "' and '" + typename(b) + "'");
}

function truediv(a, b) {
  if (typeof a === "boolean") a = a ? 1 : 0;
  if (typeof b === "boolean") b = b ? 1 : 0;
  if (__py$intish(a) && __py$intish(b)) {
    if (b === 0 || b === 0n) {
      throw pyerr("ZeroDivisionError", "division by zero");
    }
    return Number(a) / Number(b);
  }
  if (typeof a === "number" && typeof b === "number") {
    if (b === 0) throw pyerr("ZeroDivisionError", "float division by zero");
    return a / b;
  }
  throw pyerr("TypeError", "unsupported operand type(s) for /: '" +
    typename(a) + "' and '" + typename(b) + "'");
}

function floordiv(a, b) {
  if (typeof a === "boolean") a = a ? 1 : 0;
  if (typeof b === "boolean") b = b ? 1 : 0;
  if (__py$intish(a) && __py$intish(b)) {
    if (b === 0 || b === 0n) {
      throw pyerr("ZeroDivisionError",
        "integer division or modulo by zero");
    }
    if (typeof a === "bigint" || typeof b === "bigint") {
      var A = __py$bi(a), B = __py$bi(b);
      var q = A / B;
      if (q < 0n && A % B !== 0n) q -= 1n;
      return __py$n(q);
    }
    return Math.floor(a / b);
  }
  if (typeof a === "number" && typeof b === "number") {
    if (b === 0) {
      throw pyerr("ZeroDivisionError", "float floor division by zero");
    }
    return Math.floor(a / b);
  }
  throw pyerr("TypeError", "unsupported operand type(s) for //: '" +
    typename(a) + "' and '" + typename(b) + "'");
}

function pymod(a, b) {
  if (typeof a === "boolean") a = a ? 1 : 0;
  if (typeof b === "boolean") b = b ? 1 : 0;
  if (__py$intish(a) && __py$intish(b)) {
    if (b === 0 || b === 0n) {
      throw pyerr("ZeroDivisionError",
        "integer division or modulo by zero");
    }
    if (typeof a === "bigint" || typeof b === "bigint") {
      var A = __py$bi(a), B = __py$bi(b);
      var r = A % B;
      if (r !== 0n && (r < 0n) !== (B < 0n)) r += B;
      return __py$n(r);
    }
    var rn = a % b;
    if (rn !== 0 && (rn < 0) !== (b < 0)) rn += b;
    return rn;
  }
  if (typeof a === "number" && typeof b === "number") {
    if (b === 0) throw pyerr("ZeroDivisionError", "float modulo");
    var rf = a % b;
    if (rf !== 0 && (rf < 0) !== (b < 0)) rf += b;
    return rf;
  }
  throw pyerr("TypeError", "unsupported operand type(s) for %: '" +
    typename(a) + "' and '" + typename(b) + "'");
}

function pypow(a, b) {
  if (typeof a === "boolean") a = a ? 1 : 0;
  if (typeof b === "boolean") b = b ? 1 : 0;
  var ints = __py$intish(a) && __py$intish(b);
  if (ints) {
    if (typeof a === "number" && typeof b === "number") {
      if (b < 0) return Math.pow(a, b);
      if (b === 0) return 1;
      var r = 1n, base = BigInt(a), n = BigInt(b);
      while (n > 0n) {
        if (n & 1n) r *= base;
        base *= base;
        n >>= 1n;
      }
      return __py$n(r);
    }
    var A = __py$bi(a), B = __py$bi(b);
    if (B < 0n) return Math.pow(Number(A), Number(B));
    var rr = 1n, bs = A;
    while (B > 0n) {
      if (B & 1n) rr *= bs;
      bs *= bs;
      B >>= 1n;
    }
    return __py$n(rr);
  }
  if (typeof a === "number" && typeof b === "number") {
    if (a < 0 && !Number.isInteger(b)) {
      throw pyerr("ValueError",
        "negative number cannot be raised to a fractional power");
    }
    return Math.pow(a, b);
  }
  if (__py$intish(a) && typeof b === "number") {
    if (a < 0 && !Number.isInteger(b)) {
      throw pyerr("ValueError",
        "negative number cannot be raised to a fractional power");
    }
    return Math.pow(Number(a), b);
  }
  if (typeof a === "number" && typeof b === "bigint") {
    if (b < 0n) return Math.pow(a, Number(b));
    if (b > 4096n) return Math.pow(a, Number(b));
    return pypow(a, Number(b));
  }
  throw pyerr("TypeError", "unsupported operand type(s) for **: '" +
    typename(a) + "' and '" + typename(b) + "'");
}

function __py$small(v) {
  if (typeof v === "number") return v > -2147483649 && v < 2147483648;
  if (typeof v === "bigint") return v > -2147483649n && v < 2147483648n;
  if (typeof v === "boolean") return true;
  return false;
}

function pyand(a, b) {
  if (a instanceof Set && b instanceof Set) {
    var si = new $Set();
    for (var sx of a) if (b.has(sx)) si.add(sx);
    return si;
  }
  if (a instanceof $Dict && b instanceof $Dict &&
      a.__pycnt !== undefined && b.__pycnt !== undefined) {
    var res = new $Counter();
    var e, lo;
    for (e of a) {
      if (b.has(e[0])) {
        lo = b.get(e[0]);
        res.set(e[0], e[1] < lo ? e[1] : lo);
      }
    }
    return res;
  }
  if (__py$small(a) && __py$small(b)) return a & b;
  var x = __py$bi(a), y = __py$bi(b);
  if (x === null || y === null) {
    throw pyerr("TypeError", "unsupported operand type(s) for &: '" +
      typename(a) + "' and '" + typename(b) + "'");
  }
  return __py$n(x & y);
}

function pyor(a, b) {
  if (a instanceof Set && b instanceof Set) return __py$setUnion(a, b);
  if (a instanceof $Dict && b instanceof $Dict) {
    var e, d;
    if (a.__pycnt !== undefined || b.__pycnt !== undefined) {
      var res = new $Counter();
      for (e of a) res.set(e[0], e[1]);
      for (e of b) {
        if (!res.has(e[0]) || e[1] > res.get(e[0])) res.set(e[0], e[1]);
      }
      return res;
    }
    d = new $Dict();
    for (e of a) d.set(e[0], e[1]);
    for (e of b) d.set(e[0], e[1]);
    return d;
  }
  if (__py$small(a) && __py$small(b)) return a | b;
  var x = __py$bi(a), y = __py$bi(b);
  if (x === null || y === null) {
    throw pyerr("TypeError", "unsupported operand type(s) for |: '" +
      typename(a) + "' and '" + typename(b) + "'");
  }
  return __py$n(x | y);
}

function pyxor(a, b) {
  if (a instanceof Set && b instanceof Set) {
    var d = new $Set();
    for (var x of a) if (!b.has(x)) d.add(x);
    for (var y of b) if (!a.has(y)) d.add(y);
    return d;
  }
  if (__py$small(a) && __py$small(b)) return a ^ b;
  var p = __py$bi(a), q = __py$bi(b);
  if (p === null || q === null) {
    throw pyerr("TypeError", "unsupported operand type(s) for ^: '" +
      typename(a) + "' and '" + typename(b) + "'");
  }
  return __py$n(p ^ q);
}

function pyshl(a, b) {
  if (!__py$intish(a) || !__py$intish(b)) {
    throw pyerr("TypeError", "unsupported operand type(s) for <<: '" +
      typename(a) + "' and '" + typename(b) + "'");
  }
  if (typeof b === "number" ? b < 0 : b < 0n) {
    return pyshr(a, -b);
  }
  var sh = typeof b === "bigint" ? Number(b) : b;
  if (__py$intish(a) && typeof a === "number" && sh < 1024) {
    var r = a * Math.pow(2, sh);
    if (Number.isSafeInteger(r)) return r;
  }
  return __py$n(__py$bi(a) << (typeof b === "bigint" ? b : BigInt(b)));
}

function pyshr(a, b) {
  if (!__py$intish(a) || !__py$intish(b)) {
    throw pyerr("TypeError", "unsupported operand type(s) for >>: '" +
      typename(a) + "' and '" + typename(b) + "'");
  }
  if (typeof b === "number" ? b < 0 : b < 0n) {
    return pyshl(a, -b);
  }
  if (typeof a === "number" && typeof b === "number") {
    if (b > 1023) return a < 0 ? -1 : 0;
    return Math.floor(a / Math.pow(2, b));
  }
  return __py$n(__py$bi(a) >> (typeof b === "bigint" ? b : BigInt(b)));
}

function pyinvert(a) {
  if (!__py$intish(a)) {
    throw pyerr("TypeError",
      "unsupported operand type(s) for ~: '" + typename(a) + "'");
  }
  if (typeof a === "bigint") return -a - 1n;
  return -a - 1;
}

// ---------------------------------------------------------------------------
// truthiness / equality / ordering
// ---------------------------------------------------------------------------
function truthy(v) {
  if (v === null || v === undefined) return false;
  if (typeof v === "boolean") return v;
  if (typeof v === "number") return v !== 0;
  if (typeof v === "bigint") return v !== 0n;
  if (typeof v === "string") return v.length > 0;
  if (Array.isArray(v)) return v.length > 0;
  if (v instanceof Map || v instanceof Set) return v.size > 0;
  if (typeof v === "object") {
    if (typeof v.length === "number") return v.length > 0;
    return true;
  }
  return !!v;
}

function __py$namedtVals(v) {
  var outp = [];
  for (var i = 0; i < v.$fields.length; i++) outp.push(v[v.$fields[i]]);
  outp.__pytup = 1;
  return outp;
}

function eq(a, b) {
  if (a === b) return true;
  if (a !== null && typeof a === "object" && a.$namedt) {
    a = __py$namedtVals(a);
  }
  if (b !== null && typeof b === "object" && b.$namedt) {
    b = __py$namedtVals(b);
  }
  if (typeof a === "number" && typeof b === "number") return a === b;
  var ta = typeof a, tb = typeof b;
  if ((ta === "bigint" || (ta === "number" && Number.isInteger(a))) &&
      (tb === "bigint" || (tb === "number" && Number.isInteger(b)))) {
    if (ta === tb) return a === b;
    return __py$bi(a) === __py$bi(b);
  }
  if (ta === "boolean" || tb === "boolean") {
    var x = a === true ? 1 : a === false ? 0 : a;
    var y = b === true ? 1 : b === false ? 0 : b;
    return x === y;
  }
  if (a === null || a === undefined || b === null || b === undefined) {
    return false;
  }
  if (ta === "string" || tb === "string") return false;
  if (Array.isArray(a) && Array.isArray(b)) {
    if ((a.__pytup !== undefined) !== (b.__pytup !== undefined)) return false;
    if (a.length !== b.length) return false;
    for (var i = 0; i < a.length; i++) if (!eq(a[i], b[i])) return false;
    return true;
  }
  if (Array.isArray(a) || Array.isArray(b)) return false;
  if (a instanceof Map && b instanceof Map) {
    if (a.size !== b.size) return false;
    for (var e of a) {
      if (!b.has(e[0]) || !eq(e[1], b.get(e[0]))) return false;
    }
    return true;
  }
  if (a instanceof Map || b instanceof Map) return false;
  if (a instanceof Set && b instanceof Set) {
    if (a.size !== b.size) return false;
    for (var x2 of a) if (!b.has(x2)) return false;
    return true;
  }
  if (a instanceof Set || b instanceof Set) return false;
  if (a instanceof $Deque && b instanceof $Deque) {
    var la = a.length, lb = b.length;
    if (la !== lb) return false;
    var ai = a[Symbol.iterator](), bi = b[Symbol.iterator]();
    for (var i2 = 0; i2 < la; i2++) {
      if (!eq(ai.next().value, bi.next().value)) return false;
    }
    return true;
  }
  return false;
}

function cmp(a, b) {
  if (typeof a === "number" && typeof b === "number") {
    return a < b ? -1 : a > b ? 1 : 0;
  }
  if (typeof a === "string" && typeof b === "string") {
    return a < b ? -1 : a > b ? 1 : 0;
  }
  if (__py$intish(a) && __py$intish(b)) {
    return a < b ? -1 : a > b ? 1 : 0;
  }
  if ((typeof a === "boolean" || typeof b === "boolean") &&
      __py$intish(a) && __py$intish(b)) {
    return a < b ? -1 : a > b ? 1 : 0;
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    if ((a.__pytup !== undefined) !== (b.__pytup !== undefined)) {
      throw pyerr("TypeError",
        "'<' not supported between instances of '" + typename(a) +
        "' and '" + typename(b) + "'");
    }
    var n = a.length < b.length ? a.length : b.length;
    for (var i = 0; i < n; i++) {
      var c = cmp(a[i], b[i]);
      if (c !== 0) return c;
    }
    return a.length < b.length ? -1 : a.length > b.length ? 1 : 0;
  }
  if (a instanceof $Deque && b instanceof $Deque) {
    var ai = a[Symbol.iterator](), bi = b[Symbol.iterator]();
    while (true) {
      var na = ai.next(), nb = bi.next();
      if (na.done && nb.done) return 0;
      if (na.done) return -1;
      if (nb.done) return 1;
      var cc = cmp(na.value, nb.value);
      if (cc !== 0) return cc;
    }
  }
  throw pyerr("TypeError",
    "'<' not supported between instances of '" + typename(a) +
    "' and '" + typename(b) + "'");
}

function contains(h, n) {
  if (typeof h === "string") {
    if (typeof n !== "string") {
      throw pyerr("TypeError", "argument of type '" + typename(n) +
        "' is not contained in str");
    }
    return h.indexOf(n) >= 0;
  }
  if (h instanceof Map) return h.has(n);
  if (h instanceof Set) return h.has(n);
  if (h instanceof $Range) {
    if (!__py$intish(n)) return false;
    return h.__py$contains(n);
  }
  if (Array.isArray(h)) {
    for (var i = 0; i < h.length; i++) if (eq(h[i], n)) return true;
    return false;
  }
  if (h instanceof $Deque) {
    for (var x of h) if (eq(x, n)) return true;
    return false;
  }
  if (h && typeof h[Symbol.iterator] === "function") {
    for (var y of h) if (eq(y, n)) return true;
    return false;
  }
  throw pyerr("TypeError", "argument of type '" + typename(h) +
    "' is not iterable");
}

function inRange(x, s, e, k) {
  if (!__py$intish(x)) return false;
  if (k === undefined) k = 1;
  if (!__py$intish(s) || !__py$intish(e) || !__py$intish(k)) {
    throw pyerr("TypeError", "range() args must be integers");
  }
  if (k === 0 || k === 0n) {
    throw pyerr("ValueError", "range() arg 3 must not be zero");
  }
  var big = (typeof x === "bigint" || typeof s === "bigint" ||
    typeof e === "bigint" || typeof k === "bigint");
  if (big) {
    var X = __py$bi(x), S = __py$bi(s), E = __py$bi(e), K = __py$bi(k);
    if (K > 0n) {
      return X >= S && X < E && (X - S) % K === 0n;
    }
    return X <= S && X > E && (X - S) % K === 0n;
  }
  if (k > 0) return x >= s && x < e && (x - s) % k === 0;
  return x <= s && x > e && (x - s) % k === 0;
}

// ---------------------------------------------------------------------------
// canonical keys: arrays serialize (tuple keys in dicts/sets unify), other
// primitives pass through so JS Map equality matches python's ==.
// ---------------------------------------------------------------------------
function mkey(k) {
  if (typeof k === "boolean") return k ? 1 : 0;
  if (typeof k === "bigint") {
    return (k >= -9007199254740991n && k <= 9007199254740991n)
      ? Number(k) : k;
  }
  if (Array.isArray(k)) return ser(k);
  if (k instanceof Map || k instanceof Set) {
    throw pyerr("TypeError", "unhashable type: '" + typename(k) + "'");
  }
  if (k === undefined) return null;
  return k;
}

var __py$hid = 0;

function ser(v) {
  var s = "\u0000#(";
  for (var i = 0; i < v.length; i++) {
    var x = v[i];
    if (i) s += ",";
    if (x === null || x === undefined) s += "\u0000Z";
    else if (typeof x === "string") s += JSON.stringify(x);
    else if (typeof x === "number") s += "n" + (x === 0 ? 0 : x);
    else if (typeof x === "bigint") s += "n" + x;
    else if (typeof x === "boolean") s += "n" + (x ? 1 : 0);
    else if (Array.isArray(x)) s += ser(x);
    else if (x instanceof Map || x instanceof Set) {
      throw pyerr("TypeError", "unhashable type: '" + typename(x) + "'");
    } else if (typeof x === "object") {
      if (!x.$hid) x.$hid = ++__py$hid;
      s += "o" + x.$hid;
    } else s += "n" + x;
  }
  return s + ")";
}

function __py$setUnion(a, b) {
  var d = new $Set();
  for (var x of a) d.add(x);
  for (var y of b) d.add(y);
  return d;
}

// ---------------------------------------------------------------------------
// container classes
// ---------------------------------------------------------------------------
function tup() {
  var a = [];
  for (var i = 0; i < arguments.length; i++) a.push(arguments[i]);
  a.__pytup = 1;
  return a;
}

class $Dict extends Map {
  constructor() {
    super();
    this.$fact = undefined;
    this.$orig = null;
  }
  set(k, v) {
    var m = mkey(k);
    if (Array.isArray(k)) {
      if (!this.$orig) this.$orig = new Map();
      this.$orig.set(m, k);
    }
    return super.set(m, v);
  }
  get(k) { return super.get(mkey(k)); }
  has(k) { return super.has(mkey(k)); }
  delete(k) {
    if (this.$orig) this.$orig.delete(mkey(k));
    return super.delete(mkey(k));
  }
  keys() {
    var res = [];
    for (var m of super.keys()) {
      res.push((this.$orig && this.$orig.has(m)) ? this.$orig.get(m) : m);
    }
    return res;
  }
  values() {
    var res = [];
    for (var v of super.values()) res.push(v);
    return res;
  }
  items() {
    var res = [];
    for (var e of super.entries()) {
      var k = (this.$orig && this.$orig.has(e[0])) ? this.$orig.get(e[0])
        : e[0];
      var t = [k, e[1]];
      t.__pytup = 1;
      res.push(t);
    }
    return res;
  }
}

class $Counter extends $Dict {
  constructor() {
    super();
    this.__pycnt = 1;
  }
}

class $DefaultDict extends $Dict {
  constructor(fact) {
    super();
    this.$fact = fact;
  }
}

class $OrderedDict extends $Dict {
}

class $Set extends Set {
  constructor(items) {
    super();
    this.$orig = null;
    if (items !== undefined && items !== null) {
      for (var x of __py$iterable(items)) this.add(x);
    }
  }
  add(v) {
    var m = mkey(v);
    if (Array.isArray(v)) {
      if (!this.$orig) this.$orig = new Map();
      this.$orig.set(m, v);
    }
    super.add(m);
    return this;
  }
  has(v) { return super.has(mkey(v)); }
  delete(v) {
    if (this.$orig) this.$orig.delete(mkey(v));
    return super.delete(mkey(v));
  }
  *values() {
    for (var m of super.values()) {
      yield (this.$orig && this.$orig.has(m)) ? this.$orig.get(m) : m;
    }
  }
  *keys() { yield* this.values(); }
  *entries() {
    for (var v of this.values()) yield [v, v];
  }
  [Symbol.iterator]() { return this.values(); }
}

class $Deque {
  constructor(items, maxlen) {
    this.a = [];
    this.h = 0;
    this.maxlen = -1;
    if (maxlen !== undefined && maxlen !== null) {
      this.maxlen = Number(maxlen);
    }
    if (items !== undefined && items !== null) {
      for (var x of __py$iterable(items)) this.a.push(x);
      if (this.maxlen >= 0 && this.a.length > this.maxlen) {
        this.h = this.a.length - this.maxlen;
      }
    }
  }
  get length() { return this.a.length - this.h; }
  *[Symbol.iterator]() {
    for (var i = this.h; i < this.a.length; i++) yield this.a[i];
  }
  __py$trim() {
    if (this.maxlen >= 0) {
      while (this.length > this.maxlen) this.h++;
    }
    if (this.h > 64 && this.h * 2 > this.a.length) {
      this.a = this.a.slice(this.h);
      this.h = 0;
    }
  }
  __py$norm(i) {
    var n = this.length;
    if (typeof i === "boolean") i = i ? 1 : 0;
    if (typeof i === "bigint") i = Number(i);
    if (typeof i !== "number" || !Number.isInteger(i)) {
      throw pyerr("TypeError", "index must be integer, not " + typename(i));
    }
    if (i < 0) i += n;
    if (i < 0 || i >= n) throw pyerr("IndexError", "index out of range");
    return this.h + i;
  }
  get(i) { return this.a[this.__py$norm(i)]; }
  set(i, v) { this.a[this.__py$norm(i)] = v; }
  push(v) { this.a.push(v); this.__py$trim(); }
  pushLeft(v) {
    if (this.h > 0) this.a[--this.h] = v;
    else this.a.unshift(v);
    this.__py$trim();
  }
  popItem() {
    if (this.length <= 0) throw pyerr("IndexError", "pop from an empty deque");
    return this.a.pop();
  }
  popLeft() {
    if (this.length <= 0) throw pyerr("IndexError", "pop from an empty deque");
    var v = this.a[this.h++];
    this.__py$trim();
    return v;
  }
  rotate(n) {
    var len = this.length;
    if (len <= 1) return;
    if (typeof n === "bigint") n = Number(n);
    if (typeof n === "boolean") n = n ? 1 : 0;
    var r = Number(n % len);
    r = ((r % len) + len) % len;
    if (r === 0) return;
    var seg1 = this.a.slice(this.h + len - r, this.h + len);
    var seg2 = this.a.slice(this.h, this.h + len - r);
    this.a = seg1.concat(seg2);
    this.h = 0;
  }
  slice(lo, hi, st) {
    var arr = Array.from(this);
    var idx = __py$sliceIdx(arr.length, lo, hi, st);
    var count = idx.step > 0
      ? Math.max(0, Math.ceil((idx.iEnd - idx.i0) / idx.step))
      : Math.max(0, Math.ceil((idx.i0 - idx.iEnd) / (-idx.step)));
    var d = new $Deque();
    for (var j = 0; j < count; j++) d.a.push(arr[idx.i0 + j * idx.step]);
    return d;
  }
}

class $Range {
  constructor(s, e, k) {
    if (k === 0 || k === 0n) {
      throw pyerr("ValueError", "range() arg 3 must not be zero");
    }
    var big = (typeof s === "bigint" || typeof e === "bigint" ||
      typeof k === "bigint");
    if (big) {
      var bs = __py$bi(s), be = __py$bi(e), bk = __py$bi(k);
      if (bs === null || be === null || bk === null) {
        throw pyerr("TypeError", "'float' object cannot be interpreted as an integer");
      }
      s = bs; e = be; k = bk;
    } else {
      if (!Number.isInteger(s) || !Number.isInteger(e) ||
          !Number.isInteger(k)) {
        throw pyerr("TypeError",
          "'float' object cannot be interpreted as an integer");
      }
    }
    this.s = s;
    this.e = e;
    this.k = k;
  }
  get length() {
    if (typeof this.s === "bigint") {
      var d = this.k > 0n ? this.e - this.s : this.s - this.e;
      if (d <= 0n) return 0;
      var kk = this.k > 0n ? this.k : -this.k;
      return Number((d + kk - 1n) / kk);
    }
    var dd = this.k > 0 ? this.e - this.s : this.s - this.e;
    if (dd <= 0) return 0;
    var kk2 = this.k > 0 ? this.k : -this.k;
    return Math.ceil(dd / kk2);
  }
  at(n) {
    if (typeof this.s === "bigint") return this.s + BigInt(n) * this.k;
    return this.s + n * this.k;
  }
  __py$contains(x) { return inRange(x, this.s, this.e, this.k); }
  pySlice(lo, hi, st) {
    var n = this.length;
    var idx = __py$sliceIdx(n, lo, hi, st);
    var count = idx.step > 0
      ? Math.max(0, Math.ceil((idx.iEnd - idx.i0) / idx.step))
      : Math.max(0, Math.ceil((idx.i0 - idx.iEnd) / (-idx.step)));
    if (typeof this.s === "bigint") {
      var ns = this.s + BigInt(idx.i0) * this.k;
      var nk = this.k * BigInt(idx.step);
      var ne = ns + BigInt(count) * nk;
      return new $Range(ns, ne, nk);
    }
    var ns2 = this.s + idx.i0 * this.k;
    var nk2 = this.k * idx.step;
    return new $Range(ns2, ns2 + count * nk2, nk2);
  }
  *[Symbol.iterator]() {
    if (typeof this.s === "bigint") {
      if (this.k > 0n) {
        for (var i = this.s; i < this.e; i += this.k) yield i;
      } else {
        for (var j = this.s; j > this.e; j += this.k) yield j;
      }
      return;
    }
    if (this.k > 0) {
      for (var a = this.s; a < this.e; a += this.k) yield a;
    } else {
      for (var b = this.s; b > this.e; b += this.k) yield b;
    }
  }
}

// ---------------------------------------------------------------------------
// sequence / container operations
// ---------------------------------------------------------------------------
function __py$iterable(v) {
  if (v === null || v === undefined) {
    throw pyerr("TypeError", "'NoneType' object is not iterable");
  }
  if (v instanceof Map) return v.keys();
  if (typeof v === "string" || Array.isArray(v)) return v;
  if (typeof v === "object" && v.$namedt) {
    var outv = [];
    for (var fi = 0; fi < v.$fields.length; fi++) outv.push(v[v.$fields[fi]]);
    return outv;
  }
  var t = typeof v;
  if (t === "number" || t === "boolean" || t === "bigint" ||
      t === "function") {
    throw pyerr("TypeError", "'" + typename(v) + "' object is not iterable");
  }
  if (typeof v[Symbol.iterator] === "function") return v;
  throw pyerr("TypeError", "'" + typename(v) + "' object is not iterable");
}

function iter(v) { return __py$iterable(v); }

function unpack(v) {
  if (Array.isArray(v)) return v;
  if (typeof v === "string") return Array.from(v);
  if (v !== null && typeof v === "object" && v.$namedt) {
    return __py$namedtVals(v);
  }
  if (v instanceof Map) return Array.from(v.keys());
  if (v instanceof Set || v instanceof $Deque || v instanceof $Range) {
    return Array.from(v);
  }
  if (v !== null && v !== undefined &&
      typeof v[Symbol.iterator] === "function") {
    return Array.from(v);
  }
  throw pyerr("TypeError", "cannot unpack non-iterable '" + typename(v) + "'");
}

function __py$idx(i, len) {
  if (typeof i === "boolean") i = i ? 1 : 0;
  if (typeof i === "bigint") {
    if (i < 0n) i += BigInt(len);
    if (i < 0n || i >= BigInt(len)) {
      throw pyerr("IndexError", "index out of range");
    }
    return Number(i);
  }
  if (typeof i === "number") {
    if (!Number.isInteger(i)) {
      throw pyerr("TypeError",
        "list indices must be integers or slices, not float");
    }
    if (i < 0) i += len;
    if (i < 0 || i >= len) throw pyerr("IndexError", "index out of range");
    return i;
  }
  throw pyerr("TypeError", "'" + typename(i) +
    "' object cannot be interpreted as an integer");
}

function at(c, i) {
  if (c === null || c === undefined) {
    throw pyerr("TypeError", "'NoneType' object is not subscriptable");
  }
  if (typeof c === "string") return c.charAt(__py$idx(i, c.length));
  if (Array.isArray(c)) return c[__py$idx(i, c.length)];
  if (typeof c === "object" && c.$namedt) {
    return c[c.$fields[__py$idx(i, c.$fields.length)]];
  }
  if (c instanceof $Range) return c.at(__py$idx(i, c.length));
  if (c instanceof $Deque) return c.get(i);
  if (c instanceof Map) return dget(c, i);
  throw pyerr("TypeError", "'" + typename(c) + "' object is not subscriptable");
}

function __py$reprKey(k) {
  try {
    return __py$fmt(k, true);
  } catch (e) {
    return String(k);
  }
}

function dget(d, k) {
  if (!(d instanceof Map)) {
    if (d === null || d === undefined) {
      throw pyerr("TypeError", "'NoneType' object is not subscriptable");
    }
    throw pyerr("TypeError", "'" + typename(d) + "' object is not subscriptable");
  }
  if (!d.has(k)) {
    if (d.__pycnt !== undefined) return 0;
    if (d.$fact !== undefined) {
      var v = d.$fact();
      d.set(k, v);
      return v;
    }
    throw pyerr("KeyError", __py$reprKey(k));
  }
  return d.get(k);
}

function aset(c, i, v) {
  if (typeof c === "string") {
    throw pyerr("TypeError", "'str' object does not support item assignment");
  }
  if (Array.isArray(c)) {
    c[__py$idx(i, c.length)] = v;
    return;
  }
  if (c instanceof $Deque) {
    c.set(i, v);
    return;
  }
  if (c instanceof Map) {
    c.set(i, v);
    return;
  }
  throw pyerr("TypeError", "'" + typename(c) +
    "' object does not support item assignment");
}

function delitem(c, i) {
  if (Array.isArray(c)) {
    c.splice(__py$idx(i, c.length), 1);
    return;
  }
  if (c instanceof $Deque) {
    c.a.splice(c.__py$norm(i), 1);
    c.__py$trim();
    return;
  }
  if (c instanceof Map) {
    c.delete(i);
    return;
  }
  throw pyerr("TypeError", "'" + typename(c) +
    "' object doesn't support item deletion");
}

function __py$binop(a, op, rhs) {
  switch (op) {
    case "+":
      if (Array.isArray(a) && Array.isArray(rhs)) return concat(a, rhs);
      return pyadd(a, rhs);
    case "-": return pysub(a, rhs);
    case "*": return pymul(a, rhs);
    case "/": return truediv(a, rhs);
    case "//": return floordiv(a, rhs);
    case "%": return pymod(a, rhs);
    case "**": return pypow(a, rhs);
    case "&": return pyand(a, rhs);
    case "|": return pyor(a, rhs);
    case "^": return pyxor(a, rhs);
    case "<<": return pyshl(a, rhs);
    case ">>": return pyshr(a, rhs);
  }
  throw pyerr("Error", "unknown augmented operator " + op);
}

function aug(c, idx, op, rhs) {
  var cur = (c instanceof Map) ? dget(c, idx) : at(c, idx);
  if (op === "+" && Array.isArray(cur) && !cur.__pytup) {
    if (Array.isArray(rhs)) {
      for (var j = 0; j < rhs.length; j++) cur.push(rhs[j]);
      return;
    }
    if (typeof rhs === "string") {
      for (var ch of rhs) cur.push(ch);
      return;
    }
    if (rhs !== null && rhs !== undefined && typeof rhs !== "number" &&
        typeof rhs !== "boolean" && typeof rhs !== "bigint" &&
        typeof rhs !== "function") {
      for (var x of __py$iterable(rhs)) cur.push(x);
      return;
    }
  }
  var nv = __py$binop(cur, op, rhs);
  if (c instanceof Map) c.set(idx, nv);
  else aset(c, idx, nv);
}

function concat(a, b) {
  if (Array.isArray(a) && Array.isArray(b)) {
    var ta = a.__pytup !== undefined;
    if (ta !== (b.__pytup !== undefined)) {
      throw pyerr("TypeError", "can only concatenate list (not '" +
        typename(b) + "') to " + typename(a));
    }
    var r = a.concat(b);
    if (ta) r.__pytup = 1;
    return r;
  }
  if (typeof a === "string" && typeof b === "string") return a + b;
  throw pyerr("TypeError", "can only concatenate list (not '" +
    typename(b) + "') to " + typename(a));
}

function repeat(c, n) {
  if (typeof n === "bigint") n = Number(n);
  if (typeof n !== "number" || !Number.isInteger(n)) {
    throw pyerr("TypeError", "can't multiply sequence by non-int of type '" +
      typename(n) + "'");
  }
  if (typeof c === "string") return strmul(c, n);
  if (Array.isArray(c)) {
    var r = [];
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < c.length; j++) r.push(c[j]);
    }
    if (c.__pytup !== undefined) r.__pytup = 1;
    return r;
  }
  throw pyerr("TypeError", "can't multiply sequence by non-int of type '" +
    typename(c) + "'");
}

function iadd(a, rhs) {
  if (Array.isArray(a)) {
    if (a.__pytup !== undefined) return concat(a, rhs);
    if (typeof rhs === "string") {
      for (var ch of rhs) a.push(ch);
      return a;
    }
    if (Array.isArray(rhs)) {
      for (var j = 0; j < rhs.length; j++) a.push(rhs[j]);
      return a;
    }
    if (rhs !== null && rhs !== undefined && typeof rhs !== "number" &&
        typeof rhs !== "boolean" && typeof rhs !== "bigint" &&
        typeof rhs !== "function") {
      for (var x of __py$iterable(rhs)) a.push(x);
      return a;
    }
    throw pyerr("TypeError", "can only concatenate list (not '" +
      typename(rhs) + "') to list");
  }
  if (a instanceof $Dict && a.__pycnt !== undefined) {
    var e, k;
    if (rhs instanceof Map) {
      for (e of rhs) {
        k = e[0];
        var nv = a.has(k) ? pyadd(a.get(k), e[1]) : e[1];
        if (nv > 0) a.set(k, nv);
        else a.delete(k);
      }
      return a;
    }
    if (rhs !== null && rhs !== undefined && typeof rhs !== "function") {
      for (var x2 of __py$iterable(rhs)) {
        var nv2 = a.has(x2) ? pyadd(a.get(x2), 1) : 1;
        if (nv2 > 0) a.set(x2, nv2);
      }
      return a;
    }
    throw pyerr("TypeError", "'unsupported operand type(s) for +=");
  }
  throw pyerr("TypeError", "unsupported augmented assignment");
}

function __py$sliceIdx(n, lo, hi, st) {
  var step = 1;
  if (st !== undefined && st !== null) {
    if (typeof st === "bigint") st = Number(st);
    if (typeof st === "boolean") st = st ? 1 : 0;
    if (typeof st !== "number" || !Number.isInteger(st)) {
      throw pyerr("TypeError", "slice indices must be integers or None");
    }
    step = st;
    if (step === 0) throw pyerr("ValueError", "slice step cannot be zero");
  }
  function bound(v, dflt) {
    if (v === undefined || v === null) return dflt;
    if (typeof v === "bigint") v = Number(v);
    if (typeof v === "boolean") v = v ? 1 : 0;
    if (typeof v !== "number" || !Number.isInteger(v)) {
      throw pyerr("TypeError", "slice indices must be integers or None");
    }
    if (v < 0) v += n;
    if (step > 0) {
      if (v < 0) v = 0;
      if (v > n) v = n;
    } else {
      if (v < 0) v = -1;
      if (v > n) v = n - 1;
    }
    return v;
  }
  var i0 = bound(lo, step > 0 ? 0 : n - 1);
  var iEnd = bound(hi, step > 0 ? n : -1);
  return { i0: i0, iEnd: iEnd, step: step };
}

function __py$sliceCount(idx) {
  return idx.step > 0
    ? Math.max(0, Math.ceil((idx.iEnd - idx.i0) / idx.step))
    : Math.max(0, Math.ceil((idx.i0 - idx.iEnd) / (-idx.step)));
}

function slice(c, lo, hi, st) {
  if (c === null || c === undefined) {
    throw pyerr("TypeError", "'NoneType' object is not subscriptable");
  }
  if (typeof c === "string") {
    var si = __py$sliceIdx(c.length, lo, hi, st);
    var sc = __py$sliceCount(si);
    var r = "";
    for (var j = 0; j < sc; j++) r += c.charAt(si.i0 + j * si.step);
    return r;
  }
  if (Array.isArray(c)) {
    var ai = __py$sliceIdx(c.length, lo, hi, st);
    var ac = __py$sliceCount(ai);
    var res = [];
    for (var q = 0; q < ac; q++) res.push(c[ai.i0 + q * ai.step]);
    if (c.__pytup !== undefined) res.__pytup = 1;
    return res;
  }
  if (c instanceof $Range) return c.pySlice(lo, hi, st);
  if (c instanceof $Deque) return c.slice(lo, hi, st);
  throw pyerr("TypeError", "'" + typename(c) + "' object is not subscriptable");
}

function sliceAssign(c, lo, hi, st, rhs) {
  if (typeof c === "string") {
    throw pyerr("TypeError", "'str' object does not support item assignment");
  }
  var arr = null;
  if (Array.isArray(c)) {
    if (c.__pytup !== undefined) {
      throw pyerr("TypeError",
        "'tuple' object does not support item assignment");
    }
    arr = c;
  } else if (c instanceof $Deque) {
    arr = Array.from(c);
  } else {
    throw pyerr("TypeError", "'" + typename(c) +
      "' object does not support item assignment");
  }
  var idx = __py$sliceIdx(arr.length, lo, hi, st);
  var count = __py$sliceCount(idx);
  var items = (Array.isArray(rhs) || typeof rhs === "string")
    ? (typeof rhs === "string" ? Array.from(rhs) : rhs)
    : Array.from(__py$iterable(rhs));
  if (idx.step === 1) {
    var merged = arr.slice(0, idx.i0).concat(items,
      arr.slice(idx.i0 + count));
    if (c instanceof $Deque) {
      c.a = merged;
      c.h = 0;
      c.__py$trim();
      return;
    }
    arr.length = 0;
    for (var q = 0; q < merged.length; q++) arr.push(merged[q]);
    return;
  }
  if (items.length !== count) {
    throw pyerr("ValueError", "attempt to assign sequence of size " +
      items.length + " to extended slice of size " + count);
  }
  for (var j = 0; j < count; j++) arr[idx.i0 + j * idx.step] = items[j];
  if (c instanceof $Deque) {
    c.a = arr;
    c.h = 0;
  }
}

// ---------------------------------------------------------------------------
// formatting: python str/repr + the format() mini-language
// ---------------------------------------------------------------------------
function __py$padexp(s) {
  var i = s.indexOf("e");
  var mant = s.slice(0, i), e = s.slice(i + 1);
  var sign = e.charAt(0), digits = e.slice(1);
  while (digits.length < 2) digits = "0" + digits;
  return mant + "e" + sign + digits;
}

// python str()/repr() of a float value.
function __py$fstr(x) {
  if (typeof x === "bigint") return x.toString();
  if (typeof x === "boolean") return x ? "True" : "False";
  if (Number.isNaN(x)) return "nan";
  if (x === Infinity) return "inf";
  if (x === -Infinity) return "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  var ax = Math.abs(x);
  if (Number.isInteger(x) && ax < 1e16) return String(x) + ".0";
  if (ax >= 1e16 || ax < 1e-4) return __py$padexp(x.toExponential());
  return String(x);
}

function __py$escape(s) {
  var q = "'";
  if (s.indexOf("'") >= 0 && s.indexOf('"') < 0) q = '"';
  var r = "";
  for (var i = 0; i < s.length; i++) {
    var code = s.charCodeAt(i);
    var c = s.charAt(i);
    if (c === "\\") r += "\\\\";
    else if (c === q) r += "\\" + q;
    else if (c === "\n") r += "\\n";
    else if (c === "\r") r += "\\r";
    else if (c === "\t") r += "\\t";
    else if (code < 0x20 || code === 0x7f) {
      r += "\\x" + (code < 16 ? "0" : "") + code.toString(16);
    } else if (code > 0x7e) {
      if (code >= 0xd800 && code <= 0xdbff && i + 1 < s.length) {
        var nx = s.charCodeAt(i + 1);
        if (nx >= 0xdc00 && nx <= 0xdfff) {
          var cp = (code - 0xd800) * 0x400 + (nx - 0xdc00) + 0x10000;
          r += "\\U" + cp.toString(16).padStart(8, "0");
          i++;
          continue;
        }
      }
      if (code <= 0xffff) {
        r += "\\u" + code.toString(16).padStart(4, "0");
      } else {
        r += "\\U" + code.toString(16).padStart(8, "0");
      }
    } else {
      r += c;
    }
  }
  return q + r + q;
}

function __py$contStr(v) {
  if (Array.isArray(v)) {
    if (v.__pytup !== undefined) {
      var tp = [];
      for (var i = 0; i < v.length; i++) tp.push(__py$fmt(v[i], true));
      if (tp.length === 1) return "(" + tp[0] + ",)";
      return "(" + tp.join(", ") + ")";
    }
    var lp = [];
    for (var j = 0; j < v.length; j++) lp.push(__py$fmt(v[j], true));
    return "[" + lp.join(", ") + "]";
  }
  if (v instanceof $Range) {
    return "range(" + __py$fmt(v.s, false) + ", " + __py$fmt(v.e, false) +
      (v.k === 1 || v.k === 1n ? "" : ", " + __py$fmt(v.k, false)) + ")";
  }
  if (v instanceof Map) {
    var ps = [];
    var itE = typeof v.items === "function" ? v.items() : v.entries();
    for (var e of itE) {
      ps.push(__py$fmt(e[0], true) + ": " + __py$fmt(e[1], true));
    }
    var body = "{" + ps.join(", ") + "}";
    if (v.__pycnt !== undefined) {
      return ps.length === 0 ? "Counter()" : "Counter(" + body + ")";
    }
    if (v instanceof $DefaultDict) return "defaultdict(...)" .slice(0, 0) +
      "defaultdict(" + (v.$fact && v.$fact.name ? v.$fact.name : "?") +
      ", " + body + ")";
    if (v instanceof $OrderedDict) return "OrderedDict(" + body + ")";
    return body;
  }
  if (v instanceof Set) {
    if (v.size === 0) return "set()";
    var sp = [];
    for (var x of v) sp.push(__py$fmt(x, true));
    return "{" + sp.join(", ") + "}";
  }
  if (v instanceof $Deque) {
    var dp = [];
    for (var y of v) dp.push(__py$fmt(y, true));
    return "deque([" + dp.join(", ") + "])";
  }
  if (v === null || v === undefined) return "None";
  if (typeof v === "object") {
    return "<" + (v.constructor && v.constructor.name ? v.constructor.name
      : "object") + " object>";
  }
  return String(v);
}

// str-or-repr of a single value (quote -> repr for strings).
function __py$fmt(v, quote) {
  if (v === null || v === undefined) return "None";
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "bigint") return v.toString();
  if (typeof v === "number") {
    if (Number.isInteger(v)) return String(v);
    return __py$fstr(v);
  }
  if (typeof v === "string") return quote ? __py$escape(v) : v;
  if (typeof v === "function") {
    return "<function " + (v.name || "?") + ">";
  }
  return __py$contStr(v);
}

function str(x) {
  if (arguments.length === 0) return "";
  return __py$fmt(x, false);
}

// str() of a value the typer knows is a float.
function strf(x) {
  if (x === null || x === undefined) return "None";
  if (typeof x === "number" || typeof x === "bigint" ||
      typeof x === "boolean") {
    return __py$fstr(x);
  }
  return __py$fmt(x, false);
}

function repr(x) {
  if (typeof x === "string") return __py$escape(x);
  return __py$fmt(x, true);
}

function strcat(a, b) {
  if (typeof a === "string" && typeof b === "string") return a + b;
  throw pyerr("TypeError", "can only concatenate str (not '" +
    typename(b) + "') to str");
}

function pyprint(vals, mask, sep, end) {
  if (sep === undefined || sep === null) sep = " ";
  if (end === undefined || end === null) end = "\n";
  var parts = [];
  for (var i = 0; i < vals.length; i++) {
    parts.push((mask & (1 << i)) ? strf(vals[i]) : str(vals[i]));
  }
  out(parts.join(sep) + end);
}

// -- exact decimal decomposition of a value -------------------------------
// returns {sig: "<digits>", exp: n} meaning value = 0.sig * 10^exp.
function __py$dec(x) {
  if (typeof x === "bigint") {
    var neg = x < 0n;
    var s = (neg ? -x : x).toString();
    return { sig: s, exp: s.length };
  }
  if (typeof x === "boolean") return { sig: x ? "1" : "0", exp: 1 };
  var v = Math.abs(x);
  if (Number.isFinite(v) === false) return { sig: "0", exp: 1 };
  if (Number.isSafeInteger(v)) {
    var si = String(v);
    return { sig: si, exp: si.length };
  }
  var buf = new DataView(new ArrayBuffer(8));
  buf.setFloat64(0, v);
  var hi = buf.getUint32(0), lo = buf.getUint32(4);
  var ebits = (hi >>> 20) & 0x7ff;
  var mant = BigInt(hi & 0xfffff) * 4294967296n + BigInt(lo);
  var k;
  if (ebits === 0) {
    k = -1074;
  } else {
    mant |= (1n << 52n);
    k = ebits - 1075;
  }
  if (k >= 0) {
    var iv = mant << BigInt(k);
    var s2 = iv.toString();
    return { sig: s2, exp: s2.length };
  }
  var num = mant;
  var p5 = 5n;
  var kk = -k;
  while (kk > 0) {
    if (kk & 1) num *= p5;
    p5 *= p5;
    kk >>= 1;
  }
  var s3 = num.toString();
  return { sig: s3, exp: s3.length + k };
}

function __py$allZerosFrom(s, i) {
  for (var j = i; j < s.length; j++) {
    if (s.charCodeAt(j) !== 48) return false;
  }
  return true;
}

// true when the digit sequence should round up at index `cut`.
function __py$upAt(sig, cut) {
  if (cut < 0 || cut >= sig.length) return false;
  var d = sig.charCodeAt(cut) - 48;
  if (d < 5) return false;
  if (d > 5) return true;
  if (cut + 1 < sig.length && !__py$allZerosFrom(sig, cut + 1)) return true;
  var last = cut === 0 ? 0 : sig.charCodeAt(cut - 1) - 48;
  return (last % 2) === 1;
}

function __py$incDigits(s) {
  var a = s.split("");
  for (var i = a.length - 1; i >= 0; i--) {
    if (a[i] !== "9") {
      a[i] = String(Number(a[i]) + 1);
      return a.join("");
    }
    a[i] = "0";
  }
  return "1" + a.join("");
}

// digits of round(value * 10^P) -- the integer tail used by f/% formats.
function __py$fDigits(dec, P) {
  var cut = dec.exp + P;
  var sig = dec.sig;
  if (__py$allZerosFrom(sig, 0)) {
    var z0 = "";
    while (z0.length < P) z0 += "0";
    return "0" + z0;
  }
  if (cut > sig.length) {
    var z = "";
    while (z.length < cut - sig.length) z += "0";
    return sig + z;
  }
  var head = cut > 0 ? sig.slice(0, cut) : "";
  if (__py$upAt(sig, cut)) head = __py$incDigits(head);
  return head;
}

// assemble a fixed-form body (f / %) honoring zero-fill with commas.
function __py$floatAsm(dec, P, suffix, neg, sign, fill, align, width,
                       zero, comma) {
  var d = __py$fDigits(dec, P);
  var len = d.length;
  var ip, fp;
  if (len > P) {
    ip = P === 0 ? d : d.slice(0, len - P);
    fp = P === 0 ? "" : d.slice(len - P);
  } else {
    ip = "0";
    fp = "";
    var z = "";
    while (z.length < P - len) z += "0";
    fp = z + d;
  }
  var sgn = neg ? "-" : (sign === "+" ? "+" : sign === " " ? " " : "");
  var dot = P > 0 ? "." : "";
  if (zero && width !== null) {
    if (comma) {
      while (sgn.length + __py$groupLen(ip) + dot.length + fp.length +
             suffix.length < width) {
        ip = "0" + ip;
      }
      if (ip.length > 3) ip = __py$group3(ip);
      return sgn + ip + dot + fp + suffix;
    }
    var body = ip + dot + fp + suffix;
    var padn = width - sgn.length - body.length;
    if (padn > 0) {
      var z2 = "";
      while (z2.length < padn) z2 += "0";
      return sgn + z2 + body;
    }
    return sgn + body;
  }
  if (comma && ip.length > 3) ip = __py$group3(ip);
  var res = sgn + ip + dot + fp + suffix;
  if (width !== null) res = __py$padTo(res, fill, align, width);
  return res;
}

function __py$group3(s) {
  var r = "";
  var cnt = 0;
  for (var i = s.length - 1; i >= 0; i--) {
    r = s.charAt(i) + r;
    cnt++;
    if (cnt % 3 === 0 && i > 0) r = "," + r;
  }
  return r;
}

function __py$groupLen(s) {
  if (s.length <= 1) return s.length;
  return s.length + Math.floor((s.length - 1) / 3);
}

function __py$padTo(s, fill, align, width) {
  if (width === null || width === undefined) width = 0;
  var pad = width - s.length;
  if (pad <= 0) return s;
  var z = "";
  while (z.length < pad) z += fill;
  if (align === "<") return s + z;
  if (align === "^") {
    var l = pad >> 1;
    var zl = "", zr = "";
    for (var i = 0; i < pad; i++) {
      if (i < l) zl += fill; else zr += fill;
    }
    return zl + s + zr;
  }
  return z + s;
}

// -- the format() mini-language -------------------------------------------
function fmt(v, spec, conv, isFloat) {
  if (conv === 114 || conv === 97) v = repr(v);
  spec = (spec === null || spec === undefined) ? "" : String(spec);
  var i = 0;
  var fill = " ", align = null;
  if (spec.length > 0 && "<>^=".indexOf(spec.charAt(0)) >= 0) {
    align = spec.charAt(0);
    i = 1;
  } else if (spec.length > 1 && "<>^=".indexOf(spec.charAt(1)) >= 0) {
    fill = spec.charAt(0);
    align = spec.charAt(1);
    i = 2;
  }
  var sign = "";
  if (i < spec.length && "+- ".indexOf(spec.charAt(i)) >= 0) {
    sign = spec.charAt(i);
    i++;
  }
  var alt = false;
  if (i < spec.length && spec.charAt(i) === "#") {
    alt = true;
    i++;
  }
  var zero = false;
  if (i < spec.length && spec.charAt(i) === "0") {
    zero = true;
    i++;
  }
  var width = null;
  var wd = "";
  while (i < spec.length && spec.charAt(i) >= "0" &&
         spec.charAt(i) <= "9") {
    wd += spec.charAt(i);
    i++;
  }
  if (wd) width = Number(wd);
  var comma = false;
  if (i < spec.length && spec.charAt(i) === ",") {
    comma = true;
    i++;
  }
  var prec = null;
  if (i < spec.length && spec.charAt(i) === ".") {
    i++;
    var pd = "";
    while (i < spec.length && spec.charAt(i) >= "0" &&
           spec.charAt(i) <= "9") {
      pd += spec.charAt(i);
      i++;
    }
    prec = pd ? Number(pd) : 0;
  }
  var type = i < spec.length ? spec.charAt(i) : "";
  if (zero && (align === null || align === ">" || align === "^")) {
    fill = "0";
    if (align === null) align = ">";
  }
  if (typeof v === "string" || v instanceof String ||
      v === null || v === undefined || typeof v === "object" ||
      typeof v === "function") {
    return __py$strSpec("" + (v === null || v === undefined ? "" : v),
                         type, fill, align, width, prec, zero);
  }
  return __py$numSpec(v, type, sign, alt, fill, align, width, prec,
                      comma, zero, isFloat);
}

function __py$strSpec(s, type, fill, align, width, prec, zero) {
  if (type === "r" || type === "a") s = __py$escape(s);
  if (prec !== null && prec < s.length) s = s.slice(0, prec);
  if (align === null) {
    align = "<";
    if (zero && fill === "0" && width !== null) {
      // python: zero flag with strings pads on the left with zeros
      align = ">";
    }
  }
  return __py$padTo(s, fill, align, width);
}

function __py$numSpec(v, type, sign, alt, fill, align, width, prec,
                      comma, zero, isFloat) {
  var neg = false;
  if (typeof v === "bigint") neg = v < 0n;
  else if (typeof v === "boolean") neg = false;
  else neg = v < 0 || Object.is(v, -0);
  if (type === "%" && typeof v === "number") {
    // python scales the float by 100 (correctly rounded) then prints it
    return __py$floatAsm(__py$dec(v * 100), prec === null ? 6 : prec, "%",
                         neg, sign, fill, align, width, zero, comma);
  }
  if (type === "%" && typeof v === "bigint") {
    var bi = v < 0n ? -v : v;
    var db = { sig: bi.toString(), exp: bi.toString().length + 2 };
    return __py$floatAsm(db, prec === null ? 6 : prec, "%", neg, sign,
                         fill, align, width, zero, comma);
  }
  var intTypes = "dixXob";
  if ((type === "" || type === "n") && typeof v === "number" &&
      (!Number.isInteger(v) || !Number.isSafeInteger(v) || isFloat)) {
    // python format(float, "") == str(float)
    var bs = neg ? "-" : (sign === "+" ? "+" : sign === " " ? " " : "");
    bs += __py$fstr(Math.abs(v));
    if (width !== null) bs = __py$padTo(bs, fill, align, width);
    return bs;
  }
  if (type === "" || type === "n") {
    if (typeof v === "bigint" || typeof v === "boolean") type = "d";
    else if (Number.isSafeInteger(v) && !isFloat) type = "d";
    else type = "g";
  }
  if (type === "r") type = "";
  if (intTypes.indexOf(type) >= 0) {
    var ib = __py$intBody(v, type, alt, prec);
    var digits = ib.digits;
    var altPfx = ib.altPfx;
    var sgn = neg ? "-" : (sign === "+" ? "+" : sign === " " ? " " : "");
    if (zero && width !== null) {
      var target = width - sgn.length - altPfx.length;
      if (comma) {
        while (__py$groupLen(digits) < target) digits = "0" + digits;
        digits = digits.length > 3 ? __py$group3(digits) : digits;
      } else {
        while (digits.length < target) digits = "0" + digits;
      }
      return sgn + altPfx + digits;
    }
    if (comma && digits.length > 3) digits = __py$group3(digits);
    var body = sgn + altPfx + digits;
    if (width !== null) body = __py$padTo(body, fill, align, width);
    return body;
  }
  if (type === "f" || type === "F") {
    var df = __py$dec(typeof v === "number" ? Math.abs(v) :
      (typeof v === "bigint" ? (v < 0n ? -v : v) : v));
    var pf = prec === null ? 6 : prec;
    var bf = __py$floatAsm(df, pf, "", neg, sign, fill, align, width, zero,
                           comma);
    if (type === "F") bf = bf.toUpperCase();
    return bf;
  }
  if (type === "e" || type === "E" || type === "g" || type === "G") {
    var vv = typeof v === "number" ? Math.abs(v) :
      (typeof v === "bigint" ? Number(v) : Number(v));
    if (typeof v === "bigint") {
      var babs = v < 0n ? -v : v;
      vv = Number(babs);
    }
    var out;
    if (type === "e" || type === "E") {
      out = __py$expBody(vv, prec === null ? 6 : prec);
    } else {
      out = __py$genBody(vv, prec === null ? 6 : prec, alt);
    }
    if (type === "E" || type === "G") out = out.toUpperCase();
    return __py$finish(out, neg, sign, fill, align, width, zero, false, "");
  }
  // unknown presentation type: fall back to str
  return __py$strSpec(str(v), type, fill, align, width, prec, zero);
}

function __py$intBody(v, type, alt, prec) {
  var digits;
  var isBig = typeof v === "bigint";
  var val = v;
  if (typeof v === "boolean") val = v ? 1 : 0;
  if (type === "d" || type === "i") {
    if (isBig) {
      digits = (val < 0n ? -val : val).toString();
    } else if (Number.isSafeInteger(val)) {
      digits = String(Math.abs(Number(val)));
    } else {
      var dd = __py$dec(Math.abs(Number(val)));
      digits = __py$fDigits(dd, 0);
    }
    if (prec !== null && digits.length < prec) {
      var z2 = "";
      while (z2.length < prec - digits.length) z2 += "0";
      digits = z2 + digits;
    }
  } else {
    var base = type === "b" ? 2 : type === "o" ? 8 : 16;
    var n;
    if (isBig) n = val < 0n ? -val : val;
    else n = BigInt(Math.abs(Math.trunc(Number(val))));
    digits = n.toString(base);
    if (type === "X") digits = digits.toUpperCase();
    if (prec !== null && digits.length < prec) {
      var z3 = "";
      while (z3.length < prec - digits.length) z3 += "0";
      digits = z3 + digits;
    }
  }
  var altPfx = "";
  if (alt) {
    if (type === "b") altPfx = "0b";
    else if (type === "o") altPfx = "0o";
    else if (type === "x") altPfx = "0x";
    else if (type === "X") altPfx = "0X";
  }
  return { digits: digits, altPfx: altPfx };
}

function __py$fixedBody(dec, P, comma) {
  var d = __py$fDigits(dec, P);
  var len = d.length;
  var ip, fp;
  if (len > P) {
    ip = P === 0 ? d : d.slice(0, len - P);
    fp = P === 0 ? "" : d.slice(len - P);
  } else {
    ip = "0";
    fp = "";
    var z = "";
    while (z.length < P - len) z += "0";
    fp = z + d;
  }
  if (comma && ip.length > 3) ip = __py$group3(ip);
  if (P > 0) return ip + "." + fp;
  return ip;
}

function __py$expBody(x, P) {
  var dec = __py$dec(x);
  var cut = P + 1;
  var sig = dec.sig;
  var res;
  if (cut > sig.length) {
    var z = "";
    while (z.length < cut - sig.length) z += "0";
    res = sig + z;
  } else {
    var head = cut > 0 ? sig.slice(0, cut) : "";
    if (__py$upAt(sig, cut)) head = __py$incDigits(head);
    res = head;
  }
  if (res === "") res = "0";
  var E = dec.exp - cut + res.length - 1;
  if (res.length === 0) E = 0;
  var mant = res.charAt(0);
  var rest = res.slice(1);
  while (rest.length < P) rest += "0";
  if (rest.length > P) rest = rest.slice(0, P);
  if (P > 0) mant += "." + rest;
  var esign = E < 0 ? "-" : "+";
  var ed = String(Math.abs(E));
  while (ed.length < 2) ed = "0" + ed;
  return mant + "e" + esign + ed;
}

function __py$genBody(x, P, alt) {
  P = Math.max(1, P);
  var dec = __py$dec(x);
  // exponent of the value rounded to P significant digits
  var sig = dec.sig;
  var res;
  if (P > sig.length) {
    var zs = "";
    while (zs.length < P - sig.length) zs += "0";
    res = sig + zs;
  } else {
    var hd = sig.slice(0, P);
    if (__py$upAt(sig, P)) hd = __py$incDigits(hd);
    res = hd;
  }
  if (res === "") res = "0";
  var E = dec.exp - P + res.length - 1;
  var body;
  if (E < -4 || E >= P) {
    body = __py$expBody(x, P - 1);
    var ei = body.indexOf("e");
    var mant = ei >= 0 ? body.slice(0, ei) : body;
    var ex = ei >= 0 ? body.slice(ei) : "";
    if (!alt) {
      if (mant.indexOf(".") >= 0) {
        mant = mant.replace(/0+$/, "");
        if (mant.charAt(mant.length - 1) === ".") {
          mant = mant.slice(0, mant.length - 1);
        }
      }
    } else if (mant.indexOf(".") < 0) {
      mant = mant + ".";
    }
    body = mant + ex;
    return body;
  }
  var fp = Math.max(P - 1 - E, 0);
  body = __py$fixedBody(dec, fp, false);
  if (alt && body.indexOf(".") < 0) return body + ".";
  if (!alt && body.indexOf(".") >= 0) {
    body = body.replace(/0+$/, "");
    if (body.charAt(body.length - 1) === ".") {
      body = body.slice(0, body.length - 1);
    }
  }
  return body;
}

function __py$finish(body, neg, sign, fill, align, width, zero, _unused,
                      _u2) {
  var prefix = "";
  if (neg) prefix = "-";
  else if (sign === "+") prefix = "+";
  else if (sign === " ") prefix = " ";
  var total = prefix.length + body.length;
  if (width !== null && width > total) {
    if (zero && fill === "0" && prefix) {
      var pad = width - total;
      var z = "";
      while (z.length < pad) z += "0";
      return prefix + z + body;
    }
    var inner = prefix + body;
    var padded = __py$padTo(inner, fill, align, width);
    return padded;
  }
  return prefix + body;
}

// ---------------------------------------------------------------------------
// method dispatch tables (recv, args, kw) -> python method behaviour
// ---------------------------------------------------------------------------
function __py$attrErr(recv, name) {
  throw pyerr("AttributeError", "'" + typename(recv) +
    "' object has no attribute '" + name + "'");
}

function __py$normIdx(n, start, end) {
  var i = start === undefined || start === null ? 0 : Number(start);
  var j = end === undefined || end === null ? n : Number(end);
  if (i < 0) i += n;
  if (j < 0) j += n;
  if (i < 0) i = 0;
  if (j < 0) j = 0;
  if (i > n) i = n;
  if (j > n) j = n;
  return [i, j];
}

function __py$strStrip(s, chars, left, right) {
  var set = chars === undefined || chars === null ? null : new Set(chars);
  function ok(c) {
    if (set) return set.has(c);
    return c === " " || c === "\t" || c === "\n" || c === "\r" ||
      c === "\f" || c === "\v";
  }
  var i = 0, j = s.length;
  if (left) while (i < j && ok(s.charAt(i))) i++;
  if (right) while (j > i && ok(s.charAt(j - 1))) j--;
  return s.slice(i, j);
}

function __py$strTitle(s) {
  var r = "", prevAlpha = false;
  for (var ch of s) {
    if ((ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z")) {
      r += prevAlpha ? ch.toLowerCase() : ch.toUpperCase();
      prevAlpha = true;
    } else {
      r += ch;
      prevAlpha = false;
    }
  }
  return r;
}

function __py$strFormat(s, a, kwv) {
  var auto = 0;
  var outp = "";
  var i = 0;
  while (i < s.length) {
    var c = s.charAt(i);
    if (c === "{" && s.charAt(i + 1) === "{") {
      outp += "{";
      i += 2;
      continue;
    }
    if (c === "}" && s.charAt(i + 1) === "}") {
      outp += "}";
      i += 2;
      continue;
    }
    if (c === "{") {
      var j = s.indexOf("}", i + 1);
      if (j < 0) outp += s.slice(i), i = s.length;
      else {
        var body = s.slice(i + 1, j);
        i = j + 1;
        var conv = null;
        var spec = null;
        var bang = body.indexOf("!");
        var colon = body.indexOf(":");
        var cut = body.length;
        if (bang >= 0 && (colon < 0 || bang < colon)) cut = bang;
        else if (colon >= 0) cut = colon;
        var name = body.slice(0, cut);
        if (bang >= 0 && (colon < 0 || bang < colon)) {
          conv = body.slice(bang + 1, colon >= 0 ? colon : undefined);
        }
        if (colon >= 0) spec = body.slice(colon + 1);
        var value;
        if (name === "") {
          value = a[auto++];
        } else if (/^\d+$/.test(name)) {
          value = a[Number(name)];
        } else {
          if (!kwv || !(name in kwv)) throw pyerr("KeyError", name);
          value = kwv[name];
        }
        if (conv === "r" || conv === "a") value = repr(value);
        else if (conv !== null && conv !== "s") {
          throw pyerr("ValueError", "Unknown conversion '" + conv + "'");
        }
        if (spec !== null && spec !== "") {
          outp += (typeof value === "string" && conv !== null)
            ? fmt(value, spec) : fmt(value, spec, null, null);
        } else {
          outp += (conv === "r" || conv === "a") ? value : str(value);
        }
      }
      continue;
    }
    outp += c;
    i++;
  }
  return outp;
}

var STRM = {
  split: function(s, a) {
    var sep = a.length ? a[0] : undefined;
    var max = a.length > 1 ? a[1] : undefined;
    if (sep === undefined || sep === null) {
      var t = s.trim();
      if (t === "") return [];
      var ws = t.split(/\s+/);
      if (max !== undefined && max !== null && Number(max) >= 0 &&
          ws.length > Number(max) + 1) {
        var rest = ws.slice(Number(max)).join(" ");
        ws = ws.slice(0, Number(max));
        ws.push(rest);
      }
      return ws;
    }
    if (typeof sep !== "string") {
      throw pyerr("TypeError", "separator must be str or None, not " +
        typename(sep));
    }
    if (sep === "") {
      throw pyerr("ValueError", "empty separator");
    }
    var limit = max === undefined || max === null ? -1 : Number(max);
    if (limit === 0) return [s];
    var res = [], pos = 0, cnt = 0;
    while (true) {
      if (limit >= 0 && cnt >= limit) break;
      var j = s.indexOf(sep, pos);
      if (j < 0) break;
      res.push(s.slice(pos, j));
      pos = j + sep.length;
      cnt++;
    }
    res.push(s.slice(pos));
    return res;
  },
  splitlines: function(s, a) {
    var keep = a.length > 0 && a[0];
    var isEol = function(c) {
      return c === "\n" || c === "\r" || c === "\v" || c === "\f" ||
        c === "\u001c" || c === "\u001d" || c === "\u001e" ||
        c === "\u0085" || c === "\u2028" || c === "\u2029";
    };
    var res = [], i = 0, n = s.length;
    while (i < n) {
      var j = i;
      var adv = 0;
      while (j < n) {
        if (s.charAt(j) === "\r" && s.charAt(j + 1) === "\n") adv = 2;
        else if (isEol(s.charAt(j))) adv = 1;
        else { j++; continue; }
        break;
      }
      if (adv) {
        res.push(keep ? s.slice(i, j + adv) : s.slice(i, j));
        i = j + adv;
      } else {
        res.push(s.slice(i, n));
        i = n;
      }
    }
    return res;
  },
  strip: function(s, a) {
    return __py$strStrip(s, a[0], true, true);
  },
  lstrip: function(s, a) {
    return __py$strStrip(s, a[0], true, false);
  },
  rstrip: function(s, a) {
    return __py$strStrip(s, a[0], false, true);
  },
  replace: function(s, a) {
    var old = a[0], nw = a[1];
    var cnt = a.length > 2 ? Number(a[2]) : -1;
    if (typeof old !== "string" || typeof nw !== "string") {
      throw pyerr("TypeError", "replace() argument 1 must be str, not " +
        typename(old));
    }
    if (old === "") {
      var r = "";
      var n = s.length;
      var done = 0;
      for (var i2 = 0; i2 <= n; i2++) {
        if (cnt >= 0 && done >= cnt) { r += s.slice(i2); break; }
        r += nw;
        done++;
        if (i2 < n) r += s.charAt(i2);
      }
      return r;
    }
    if (cnt === 0) return s;
    var outp = "", pos = 0, made = 0;
    while (cnt < 0 || made < cnt) {
      var j = s.indexOf(old, pos);
      if (j < 0) break;
      outp += s.slice(pos, j) + nw;
      pos = j + old.length;
      made++;
    }
    return outp + s.slice(pos);
  },
  format: function(s, a, kwv) {
    return __py$strFormat(s, a, kwv);
  },
  translate: function(s, a) {
    var table = a[0];
    if (table === null || table === undefined) {
      throw pyerr("TypeError", "translate() argument must be str, not None");
    }
    var r = "";
    if (table instanceof Map) {
      for (var i = 0; i < s.length; i++) {
        var ch = s.charAt(i);
        var rep = table.has(ch) ? table.get(ch) : undefined;
        if (rep === undefined) r += ch;
        else if (rep !== null) r += rep;
      }
      return r;
    }
    if (typeof table === "object") {
      for (var i2 = 0; i2 < s.length; i2++) {
        var c2 = s.charAt(i2);
        var k2 = String(s.codePointAt(i2));
        var rep2 = k2 in table ? table[k2] : undefined;
        if (rep2 === undefined) r += c2;
        else if (rep2 !== null) r += rep2;
      }
      return r;
    }
    throw pyerr("TypeError", "translate() table must be mapping");
  },
  lower: function(s) { return s.toLowerCase(); },
  upper: function(s) { return s.toUpperCase(); },
  casefold: function(s) { return s.toLowerCase(); },
  capitalize: function(s) {
    if (s === "") return s;
    return s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
  },
  title: function(s) { return __py$strTitle(s); },
  swapcase: function(s) {
    var r = "";
    for (var i = 0; i < s.length; i++) {
      var c = s.charAt(i);
      var u = c.toUpperCase(), l = c.toLowerCase();
      r += c === u ? l : u;
    }
    return r;
  },
  zfill: function(s, a) {
    var w = Number(a[0]);
    if (w <= s.length) return s;
    var pad = w - s.length;
    var sign = (s.charAt(0) === "+" || s.charAt(0) === "-")
      ? s.charAt(0) : "";
    var z = "";
    while (z.length < pad) z += "0";
    return sign + z + s.slice(sign.length);
  },
  ljust: function(s, a) {
    return __py$padTo(s, a.length > 1 ? String(a[1]) : " ", "<",
      Number(a[0]));
  },
  rjust: function(s, a) {
    return __py$padTo(s, a.length > 1 ? String(a[1]) : " ", ">",
      Number(a[0]));
  },
  center: function(s, a) {
    var w = Number(a[0]);
    var f = a.length > 1 ? String(a[1]) : " ";
    if (f.length !== 1) {
      throw pyerr("ValueError", "fill character must be exactly one " +
        "character long");
    }
    if (w <= s.length) return s;
    var pad = w - s.length;
    var left = (pad + (w % 2 === 1 ? 1 : 0)) >> 1;
    var z1 = "";
    while (z1.length < left) z1 += f;
    var z2 = "";
    while (z2.length < pad - left) z2 += f;
    return z1 + s + z2;
  },
  expandtabs: function(s, a) {
    var ts = a.length ? Number(a[0]) : 8;
    if (ts <= 0) ts = 8;
    var r = "", col = 0;
    for (var i = 0; i < s.length; i++) {
      var c = s.charAt(i);
      if (c === "\t") {
        var adv = ts - (col % ts);
        var z = "";
        while (z.length < adv) z += " ";
        r += z;
        col += adv;
      } else {
        r += c;
        if (c === "\n") col = 0;
        else col++;
      }
    }
    return r;
  },
  removeprefix: function(s, a) {
    var p = a[0];
    if (typeof p !== "string") {
      throw pyerr("TypeError", "removeprefix: expected str, got " +
        typename(p));
    }
    return s.startsWith(p) ? s.slice(p.length) : s;
  },
  removesuffix: function(s, a) {
    var p = a[0];
    if (typeof p !== "string") {
      throw pyerr("TypeError", "removesuffix: expected str, got " +
        typename(p));
    }
    return s.endsWith(p) ? s.slice(0, s.length - p.length) : s;
  },
  find: function(s, a) {
    var sub = a[0];
    if (typeof sub !== "string") {
      throw pyerr("TypeError", "must be str, not " + typename(sub));
    }
    var bd = __py$sliceIdx(s.length, a[1], a[2], undefined);
    if (bd.i0 > bd.iEnd) return -1;
    var loc = s.slice(bd.i0, bd.iEnd).indexOf(sub);
    return loc < 0 ? -1 : bd.i0 + loc;
  },
  rfind: function(s, a) {
    var sub = a[0];
    if (typeof sub !== "string") {
      throw pyerr("TypeError", "must be str, not " + typename(sub));
    }
    var bd = __py$sliceIdx(s.length, a[1], a[2], undefined);
    if (bd.i0 > bd.iEnd) return -1;
    var loc = s.slice(bd.i0, bd.iEnd).lastIndexOf(sub);
    return loc < 0 ? -1 : bd.i0 + loc;
  },
  index: function(s, a) {
    var r = STRM.find(s, a);
    if (r < 0) throw pyerr("ValueError", "substring not found");
    return r;
  },
  rindex: function(s, a) {
    var r = STRM.rfind(s, a);
    if (r < 0) throw pyerr("ValueError", "substring not found");
    return r;
  },
  count: function(s, a) {
    var sub = a[0];
    if (typeof sub !== "string") {
      throw pyerr("TypeError", "must be str, not " + typename(sub));
    }
    var bd = __py$sliceIdx(s.length, a[1], a[2], undefined);
    if (bd.i0 > bd.iEnd) return 0;
    if (sub === "") return bd.iEnd - bd.i0 + 1;
    var seg = s.slice(bd.i0, bd.iEnd);
    var cnt = 0, pos = 0;
    while (true) {
      var j = seg.indexOf(sub, pos);
      if (j < 0) break;
      cnt++;
      pos = j + sub.length;
    }
    return cnt;
  },
  join: function(s, a) {
    var it = a[0];
    var items = (typeof it === "string") ? Array.from(it)
      : (Array.isArray(it) ? it : Array.from(__py$iterable(it)));
    var parts = [];
    for (var i = 0; i < items.length; i++) {
      if (typeof items[i] !== "string") {
        throw pyerr("TypeError", "sequence item " + i + ": expected str " +
          "instance, " + typename(items[i]) + " found");
      }
      parts.push(items[i]);
    }
    return parts.join(s);
  },
  startswith: function(s, a) {
    var p = a[0];
    var bd = __py$sliceIdx(s.length, a[1], a[2], undefined);
    var seg = s.slice(bd.i0, bd.iEnd);
    if (Array.isArray(p)) {
      for (var i = 0; i < p.length; i++) {
        if (seg.startsWith(String(p[i]))) return true;
      }
      return false;
    }
    return seg.startsWith(String(p));
  },
  endswith: function(s, a) {
    var p = a[0];
    var bd = __py$sliceIdx(s.length, a[1], a[2], undefined);
    var seg = s.slice(bd.i0, bd.iEnd);
    if (Array.isArray(p)) {
      for (var i = 0; i < p.length; i++) {
        if (seg.endsWith(String(p[i]))) return true;
      }
      return false;
    }
    return seg.endsWith(String(p));
  },
  isdigit: function(s) { return s.length > 0 && /^\d+$/.test(s); },
  isdecimal: function(s) { return s.length > 0 && /^\d+$/.test(s); },
  isnumeric: function(s) { return s.length > 0 && /^\d+$/.test(s); },
  isalpha: function(s) { return s.length > 0 && /^[a-zA-Z]+$/.test(s); },
  isalnum: function(s) { return s.length > 0 && /^[a-zA-Z0-9]+$/.test(s); },
  isspace: function(s) {
    return s.length > 0 && /^[\s]+$/.test(s);
  },
  islower: function(s) {
    var cased = false;
    for (var i = 0; i < s.length; i++) {
      var c = s.charAt(i);
      if (c >= "a" && c <= "z") cased = true;
      else if (c >= "A" && c <= "Z") return false;
    }
    return cased;
  },
  isupper: function(s) {
    var cased = false;
    for (var i = 0; i < s.length; i++) {
      var c = s.charAt(i);
      if (c >= "A" && c <= "Z") cased = true;
      else if (c >= "a" && c <= "z") return false;
    }
    return cased;
  },
  istitle: function(s) {
    var cased = false;
    for (var i = 0; i < s.length; i++) {
      var c = s.charAt(i);
      if ((c >= "a" && c <= "z") || (c >= "A" && c <= "Z")) {
        cased = true;
        break;
      }
    }
    return cased && __py$strTitle(s) === s;
  },
  partition: function(s, a) {
    var sep = a[0];
    if (typeof sep !== "string") {
      throw pyerr("TypeError", "expected str instance, not " +
        typename(sep));
    }
    if (sep === "") throw pyerr("ValueError", "empty separator");
    var i = s.indexOf(sep);
    if (i < 0) return tup("", "", s);
    return tup(s.slice(0, i), sep, s.slice(i + sep.length));
  },
  rpartition: function(s, a) {
    var sep = a[0];
    if (typeof sep !== "string") {
      throw pyerr("TypeError", "expected str instance, not " +
        typename(sep));
    }
    if (sep === "") throw pyerr("ValueError", "empty separator");
    var i = s.lastIndexOf(sep);
    if (i < 0) return tup("", "", s);
    return tup(s.slice(0, i), sep, s.slice(i + sep.length));
  },
  encode: function(s) { return s; },
  decode: function(s) { return s; }
};

var TUPM = {
  index: function(c, a) { return LISTM.index(c, a); },
  count: function(c, a) { return LISTM.count(c, a); }
};

function __py$listItems(it) {
  if (typeof it === "string") return Array.from(it);
  if (Array.isArray(it)) return it.slice();
  return Array.from(__py$iterable(it));
}

var LISTM = {
  pop: function(c, a) {
    if (c.__pytup !== undefined) __py$attrErr(c, "pop");
    if (c.length === 0) throw pyerr("IndexError", "pop from empty list");
    var i = a.length > 0 ? a[0] : c.length - 1;
    if (typeof i === "bigint") i = Number(i);
    if (typeof i === "boolean") i = i ? 1 : 0;
    if (typeof i !== "number" || !Number.isInteger(i)) {
      throw pyerr("TypeError", "list indices must be integers or slices, " +
        "not " + typename(i));
    }
    if (i < 0) i += c.length;
    if (i < 0 || i >= c.length) {
      throw pyerr("IndexError", "pop index out of range");
    }
    return c.splice(i, 1)[0];
  },
  index: function(c, a) {
    var x = a[0];
    var bd = __py$sliceIdx(c.length, a[1], a[2], undefined);
    if (bd.i0 <= bd.iEnd) {
      for (var i = bd.i0; i < bd.iEnd; i++) {
        if (eq(c[i], x)) return i;
      }
    }
    throw pyerr("ValueError", repr(x) + " is not in list");
  },
  count: function(c, a) {
    var x = a[0], cnt = 0;
    for (var i = 0; i < c.length; i++) if (eq(c[i], x)) cnt++;
    return cnt;
  },
  copy: function(c) {
    if (c.__pytup !== undefined) __py$attrErr(c, "copy");
    return c.slice();
  },
  append: function(c, a) {
    if (c.__pytup !== undefined) __py$attrErr(c, "append");
    c.push(a[0]);
    return undefined;
  },
  extend: function(c, a) {
    if (c.__pytup !== undefined) __py$attrErr(c, "extend");
    var items = __py$listItems(a[0]);
    for (var i = 0; i < items.length; i++) c.push(items[i]);
    return undefined;
  },
  insert: function(c, a) {
    if (c.__pytup !== undefined) __py$attrErr(c, "insert");
    var i = a[0];
    if (typeof i === "bigint") i = Number(i);
    if (typeof i === "boolean") i = i ? 1 : 0;
    if (typeof i !== "number" || !Number.isInteger(i)) {
      throw pyerr("TypeError", "'" + typename(i) + "' object cannot be " +
        "interpreted as an integer");
    }
    var n = c.length;
    if (i < 0) {
      i += n;
      if (i < 0) i = 0;
    }
    if (i > n) i = n;
    c.splice(i, 0, a[1]);
    return undefined;
  },
  remove: function(c, a) {
    if (c.__pytup !== undefined) __py$attrErr(c, "remove");
    var x = a[0];
    for (var i = 0; i < c.length; i++) {
      if (eq(c[i], x)) {
        c.splice(i, 1);
        return undefined;
      }
    }
    throw pyerr("ValueError", "list.remove(x): x not in list");
  },
  sort: function(c, a, kwv) {
    if (c.__pytup !== undefined) __py$attrErr(c, "sort");
    var key = kwv && kwv.key !== undefined ? kwv.key : null;
    var rev = kwv && kwv.reverse ? true : false;
    var pairs = new Array(c.length);
    for (var i = 0; i < c.length; i++) {
      pairs[i] = [key ? key(c[i]) : c[i], c[i], i];
    }
    pairs.sort(function(p, q) {
      var r = cmp(p[0], q[0]);
      if (r !== 0) return rev ? -r : r;
      return p[2] - q[2];
    });
    for (var j = 0; j < pairs.length; j++) c[j] = pairs[j][1];
    return undefined;
  },
  reverse: function(c) {
    if (c.__pytup !== undefined) __py$attrErr(c, "reverse");
    c.reverse();
    return undefined;
  },
  clear: function(c) {
    if (c.__pytup !== undefined) __py$attrErr(c, "clear");
    c.length = 0;
    return undefined;
  }
};

var DICTM = {
  items: function(d) {
    if (typeof d.items === "function") return d.items();
    var res = [];
    for (var e of d.entries()) {
      var t = [e[0], e[1]];
      t.__pytup = 1;
      res.push(t);
    }
    return res;
  },
  keys: function(d) {
    if (typeof d.keys === "function" && d.keys !== Map.prototype.keys) {
      return d.keys();
    }
    return Array.from(d.keys());
  },
  values: function(d) {
    if (typeof d.values === "function" && d.values !== Map.prototype.values) {
      return d.values();
    }
    return Array.from(d.values());
  },
  get: function(d, a) {
    var k = a[0];
    var def = a.length > 1 ? a[1] : undefined;
    if (d.has(k)) return d.get(k);
    return def;
  },
  pop: function(d, a) {
    var k = a[0];
    if (d.has(k)) {
      var v = d.get(k);
      d.delete(k);
      return v;
    }
    if (a.length > 1) return a[1];
    throw pyerr("KeyError", __py$reprKey(k));
  },
  setdefault: function(d, a) {
    var k = a[0];
    var def = a.length > 1 ? a[1] : undefined;
    if (d.has(k)) return d.get(k);
    d.set(k, def);
    return def;
  },
  update: function(d, a, kwv) {
    var add = d.__pycnt !== undefined;
    var apply = function(k, v) {
      if (add) {
        var cur = d.has(k) ? d.get(k) : 0;
        d.set(k, pyadd(cur, v));
      } else {
        d.set(k, v);
      }
    };
    if (a.length > 0 && a[0] !== null && a[0] !== undefined) {
      var other = a[0];
      if (other instanceof Map) {
        for (var e of other.entries()) apply(e[0], e[1]);
      } else if (add) {
        for (var it2 of __py$iterable(other)) apply(it2, 1);
      } else {
        var pairs = __py$listItems(other);
        for (var i = 0; i < pairs.length; i++) {
          var p = pairs[i];
          if (Array.isArray(p) && p.length === 2) apply(p[0], p[1]);
          else {
            throw pyerr("TypeError", "dictionary update sequence element " +
              i + " has length " + (Array.isArray(p) ? p.length : 1) +
              "; 2 is required");
          }
        }
      }
    }
    if (kwv) {
      for (var k of Object.keys(kwv)) {
        if (k === "__pykw") continue;
        apply(k, kwv[k]);
      }
    }
    return undefined;
  },
  clear: function(d) {
    d.clear();
    return undefined;
  },
  copy: function(d) {
    var r;
    if (d.__pycnt !== undefined) r = new $Counter();
    else if (d instanceof $DefaultDict) r = new $DefaultDict(d.$fact);
    else if (d instanceof $OrderedDict) r = new $OrderedDict();
    else r = new $Dict();
    var ks = d.keys();
    for (var i = 0; i < ks.length; i++) r.set(ks[i], d.get(ks[i]));
    return r;
  },
  popitem: function(d) {
    var ks = d.keys();
    if (ks.length === 0) {
      throw pyerr("KeyError", "popitem(): dictionary is empty");
    }
    var k = ks[ks.length - 1];
    var v = d.get(k);
    d.delete(k);
    var t = [k, v];
    t.__pytup = 1;
    return t;
  },
  most_common: function(d, a) {
    if (d.__pycnt === undefined) __py$attrErr(d, "most_common");
    var n = a.length > 0 && a[0] !== null && a[0] !== undefined
      ? Number(a[0]) : null;
    var ks = d.keys();
    var pairs = [];
    for (var i = 0; i < ks.length; i++) {
      pairs.push([ks[i], d.get(ks[i]), i]);
    }
    pairs.sort(function(p, q) {
      var r = cmp(q[1], p[1]);
      if (r !== 0) return r;
      return p[2] - q[2];
    });
    if (n !== null && n < pairs.length) pairs = pairs.slice(0, Math.max(0, n));
    var res = [];
    for (var j = 0; j < pairs.length; j++) {
      var t = [pairs[j][0], pairs[j][1]];
      t.__pytup = 1;
      res.push(t);
    }
    return res;
  },
  total: function(d) {
    if (d.__pycnt === undefined) __py$attrErr(d, "total");
    var s = 0;
    for (var v of d.values()) s = pyadd(s, v);
    return s;
  },
  elements: function(d) {
    if (d.__pycnt === undefined) __py$attrErr(d, "elements");
    var res = [];
    var ks = d.keys();
    for (var i = 0; i < ks.length; i++) {
      var c = d.get(ks[i]);
      if (typeof c === "bigint" || Number(c) > 0) {
        var n = Number(c);
        for (var j = 0; j < n; j++) res.push(ks[i]);
      }
    }
    return res;
  }
};

var SETM = {
  pop: function(s) {
    if (s.size === 0) throw pyerr("KeyError", "pop from an empty set");
    var first = undefined;
    for (var v of s) {
      first = v;
      break;
    }
    s.delete(first);
    return first;
  },
  add: function(s, a) {
    s.add(a[0]);
    return undefined;
  },
  discard: function(s, a) {
    s.delete(a[0]);
    return undefined;
  },
  remove: function(s, a) {
    if (!s.has(a[0])) throw pyerr("KeyError", __py$reprKey(a[0]));
    s.delete(a[0]);
    return undefined;
  },
  clear: function(s) {
    s.clear();
    return undefined;
  },
  copy: function(s) {
    return new $Set(Array.from(s));
  },
  update: function(s, a) {
    for (var i = 0; i < a.length; i++) {
      for (var x of __py$iterable(a[i])) s.add(x);
    }
    return undefined;
  },
  union: function(s, a) {
    var r = new $Set(Array.from(s));
    for (var i = 0; i < a.length; i++) {
      for (var x of __py$iterable(a[i])) r.add(x);
    }
    return r;
  },
  intersection: function(s, a) {
    if (a.length === 0) return new $Set(Array.from(s));
    var others = [];
    for (var i = 0; i < a.length; i++) {
      var o = new Set();
      for (var x of __py$iterable(a[i])) o.add(mkey(x));
      others.push(o);
    }
    var r = new $Set();
    for (var y of s) {
      var ok = true;
      for (var j = 0; j < others.length; j++) {
        if (!others[j].has(mkey(y))) {
          ok = false;
          break;
        }
      }
      if (ok) r.add(y);
    }
    return r;
  },
  difference: function(s, a) {
    var drop = new Set();
    for (var i = 0; i < a.length; i++) {
      for (var x of __py$iterable(a[i])) drop.add(mkey(x));
    }
    var r = new $Set();
    for (var y of s) if (!drop.has(mkey(y))) r.add(y);
    return r;
  },
  symmetric_difference: function(s, a) {
    var other = new $Set();
    for (var x of __py$iterable(a[0])) other.add(x);
    var r = new $Set();
    for (var y of s) if (!other.has(y)) r.add(y);
    for (var z of other) if (!s.has(z)) r.add(z);
    return r;
  },
  isdisjoint: function(s, a) {
    var o = new Set();
    for (var x of __py$iterable(a.length > 0 ? a[0] : [])) o.add(mkey(x));
    for (var y of s) if (o.has(mkey(y))) return false;
    return true;
  },
  issubset: function(s, a) {
    var o = new Set();
    for (var x of __py$iterable(a.length > 0 ? a[0] : [])) o.add(mkey(x));
    for (var y of s) if (!o.has(mkey(y))) return false;
    return true;
  },
  issuperset: function(s, a) {
    for (var x of __py$iterable(a.length > 0 ? a[0] : [])) {
      if (!s.has(x)) return false;
    }
    return true;
  }
};

var DQM = {
  append: function(q, a) {
    q.push(a[0]);
    return undefined;
  },
  appendleft: function(q, a) {
    q.pushLeft(a[0]);
    return undefined;
  },
  pop: function(q) {
    return q.popItem();
  },
  popleft: function(q) {
    return q.popLeft();
  },
  extend: function(q, a) {
    for (var x of __py$iterable(a[0])) q.push(x);
    return undefined;
  },
  extendleft: function(q, a) {
    for (var x of __py$iterable(a[0])) q.pushLeft(x);
    return undefined;
  },
  rotate: function(q, a) {
    q.rotate(a.length > 0 ? a[0] : 0);
    return undefined;
  },
  clear: function(q) {
    q.a = [];
    q.h = 0;
    return undefined;
  },
  copy: function(q) {
    var r = new $Deque();
    for (var i = 0; i < q.length; i++) r.push(q.get(i));
    return r;
  },
  count: function(q, a) {
    var x = a[0], cnt = 0;
    for (var i = 0; i < q.length; i++) if (eq(q.get(i), x)) cnt++;
    return cnt;
  },
  index: function(q, a) {
    var x = a[0];
    for (var i = 0; i < q.length; i++) if (eq(q.get(i), x)) return i;
    throw pyerr("ValueError", repr(x) + " is not in deque");
  }
};

var RANGEM = {
  index: function(r, a) {
    var x = a[0];
    if (!r.__py$contains(x)) {
      throw pyerr("ValueError", repr(x) + " is not in range");
    }
    var num = pysub(x, r.s);
    if (typeof num === "bigint" || typeof r.k === "bigint") {
      var nb = typeof num === "bigint" ? num : BigInt(num);
      var kb = typeof r.k === "bigint" ? r.k : BigInt(r.k);
      return __py$n(nb / kb);
    }
    return num / r.k;
  },
  count: function(r, a) {
    return r.__py$contains(a[0]) ? 1 : 0;
  }
};

var NUMM = {
  bit_length: function(v) {
    if (typeof v === "boolean") v = v ? 1 : 0;
    if (typeof v === "number") {
      if (!Number.isInteger(v)) {
        throw pyerr("TypeError", "not an integer");
      }
      v = BigInt(v);
    }
    var x = v < 0n ? -v : v;
    var bits = 0;
    while (x > 0n) {
      x >>= 1n;
      bits++;
    }
    return bits;
  },
  to_bytes: function(v, a) {
    if (typeof v === "boolean") v = v ? 1 : 0;
    if (typeof v === "number" && !Number.isInteger(v)) {
      throw pyerr("TypeError", "can't convert non-integer to bytes");
    }
    var len = a[0];
    if (len === undefined || len === null) {
      throw pyerr("TypeError", "'length' parameter required");
    }
    if (typeof len === "bigint") len = Number(len);
    var bo = a.length > 1 && a[1] !== undefined ? a[1] : "big";
    if (bo !== "big" && bo !== "little") {
      throw pyerr("TypeError", "'byteorder' must be 'big' or 'little'");
    }
    var signed = a.length > 2 && a[2] ? true : false;
    var bi = typeof v === "bigint" ? v : BigInt(Math.trunc(v));
    var neg = bi < 0n;
    if (neg && !signed) {
      throw pyerr("OverflowError", "can't convert negative value to " +
        "unsigned integer");
    }
    var mod = 1n << BigInt(len * 8);
    if (neg) bi = bi + mod;
    if (bi >= mod && !neg) {
      throw pyerr("OverflowError", "int too large to convert");
    }
    if (bi < 0n || bi >= mod) {
      throw pyerr("OverflowError", "int too big to convert");
    }
    var bytes = [];
    for (var i = 0; i < len; i++) {
      bytes.push(Number((bi >> BigInt(8 * i)) & 0xffn));
    }
    if (bo === "big") bytes.reverse();
    return bytes;
  },
  is_integer: function(v) {
    if (typeof v === "bigint" || typeof v === "boolean") return true;
    return Number.isInteger(v);
  },
  as_integer_ratio: function(v) {
    if (typeof v === "bigint") return [v, 1n];
    if (typeof v === "boolean") return [v ? 1 : 0, 1];
    if (!Number.isFinite(v)) {
      throw pyerr("OverflowError", "cannot convert Infinity or NaN to " +
        "integer ratio");
    }
    if (Number.isInteger(v)) return [v, 1];
    var buf = new DataView(new ArrayBuffer(8));
    buf.setFloat64(0, Math.abs(v));
    var hi = buf.getUint32(0), lo = buf.getUint32(4);
    var ebits = (hi >>> 20) & 0x7ff;
    var mant = BigInt(hi & 0xfffff) * 4294967296n + BigInt(lo);
    var k;
    if (ebits === 0) {
      k = -1074;
    } else {
      mant |= (1n << 52n);
      k = ebits - 1075;
    }
    var num, den;
    if (k >= 0) {
      num = mant << BigInt(k);
      den = 1n;
    } else {
      num = mant;
      den = 1n << BigInt(-k);
    }
    while ((num & 1n) === 0n && (den & 1n) === 0n) {
      num >>= 1n;
      den >>= 1n;
    }
    if (v < 0) num = -num;
    return [num, den];
  }
};

function __py$hasMeth(r, n) {
  if (typeof r === "string") return Object.prototype.hasOwnProperty.call(STRM, n);
  if (typeof r === "number" || typeof r === "bigint" ||
      typeof r === "boolean") {
    return Object.prototype.hasOwnProperty.call(NUMM, n);
  }
  if (Array.isArray(r)) {
    return Object.prototype.hasOwnProperty.call(
      r.__pytup !== undefined ? TUPM : LISTM, n);
  }
  if (r instanceof $Range) return Object.prototype.hasOwnProperty.call(RANGEM, n);
  if (r instanceof $Deque) return Object.prototype.hasOwnProperty.call(DQM, n);
  if (r instanceof Map) return Object.prototype.hasOwnProperty.call(DICTM, n);
  if (r instanceof Set) return Object.prototype.hasOwnProperty.call(SETM, n);
  if (typeof r === "object") {
    return Object.prototype.hasOwnProperty.call(r, n);
  }
  return false;
}

function __py$tableFor(r) {
  if (typeof r === "string") return STRM;
  if (typeof r === "number" || typeof r === "bigint" ||
      typeof r === "boolean") {
    return NUMM;
  }
  if (Array.isArray(r)) {
    return r.__pytup !== undefined ? TUPM : LISTM;
  }
  if (r instanceof $Range) return RANGEM;
  if (r instanceof $Deque) return DQM;
  if (r instanceof Map) return DICTM;
  if (r instanceof Set) return SETM;
  return null;
}

function __py$popKw(args) {
  if (args.length > 0) {
    var last = args[args.length - 1];
    if (last !== null && typeof last === "object" && last.__pykw) {
      return [args.slice(0, args.length - 1), last];
    }
  }
  return [args, null];
}

function callm(recv, name, args) {
  var pk = __py$popKw(args);
  args = pk[0];
  var kwv = pk[1];
  if (recv === null || recv === undefined) __py$attrErr(recv, name);
  var table = __py$tableFor(recv);
  if (table && Object.prototype.hasOwnProperty.call(table, name)) {
    return table[name](recv, args, kwv);
  }
  if (table !== null) __py$attrErr(recv, name);
  if (typeof recv === "object") {
    if (recv.$fields && recv.$fields.indexOf(name) >= 0) return recv[name];
    if (Object.prototype.hasOwnProperty.call(recv, name) &&
        typeof recv[name] === "function") {
      return recv[name].apply(recv, args);
    }
  }
  __py$attrErr(recv, name);
}

function bound(r, n) {
  var table = __py$tableFor(r);
  if (table) {
    if (!Object.prototype.hasOwnProperty.call(table, n)) {
      __py$attrErr(r, n);
    }
    return function() {
      var args = Array.prototype.slice.call(arguments);
      var pk = __py$popKw(args);
      return table[n](r, pk[0], pk[1]);
    };
  }
  if (r === null || r === undefined) __py$attrErr(r, n);
  if (typeof r === "object") {
    if (r.$fields && r.$fields.indexOf(n) >= 0) return r[n];
    if (Object.prototype.hasOwnProperty.call(r, n)) {
      var f = r[n];
      if (typeof f === "function") {
        return function() {
          var pk = __py$popKw(Array.prototype.slice.call(arguments));
          return f.apply(r, pk[0]);
        };
      }
      return f;
    }
  }
  __py$attrErr(r, n);
}

function kw(o) {
  if (o !== null && typeof o === "object") o.__pykw = true;
  return o;
}

// ---------------------------------------------------------------------------
// builtins (bare names exactly as the backend emits them)
// ---------------------------------------------------------------------------
function len(v) {
  if (v === null || v === undefined) {
    throw pyerr("AttributeError",
      "'NoneType' object has no attribute '__len__'");
  }
  if (typeof v === "string") return v.length;
  if (Array.isArray(v)) return v.length;
  if (v instanceof Map || v instanceof Set) return v.size;
  if (v instanceof $Deque || v instanceof $Range) return v.length;
  if (v instanceof $Dict) return v.size;
  if (typeof v === "object" && v.$namedt) return v.$fields.length;
  if (typeof v === "number" || typeof v === "boolean" ||
      typeof v === "bigint") {
    throw pyerr("TypeError",
      "object of type '" + typename(v) + "' has no len()");
  }
  if (v !== null && typeof v === "object" && typeof v.length === "number") {
    return v.length;
  }
  throw pyerr("TypeError",
    "object of type '" + typename(v) + "' has no len()");
}

function list(v) {
  if (v === undefined) return [];
  if (v === null) {
    throw pyerr("TypeError", "'NoneType' object is not iterable");
  }
  if (v instanceof Map) {
    var ks = typeof v.keys === "function" && v.keys !== Map.prototype.keys
      ? v.keys() : Array.from(v.keys());
    return ks.slice ? ks.slice() : Array.from(ks);
  }
  return __py$listItems(v);
}

function tuple(v) {
  if (v === undefined) return tup();
  if (v === null) {
    throw pyerr("TypeError", "'NoneType' object is not iterable");
  }
  var a;
  if (Array.isArray(v)) a = v.slice();
  else if (typeof v === "string") a = Array.from(v);
  else if (v instanceof Map) {
    var ks = typeof v.keys === "function" && v.keys !== Map.prototype.keys
      ? v.keys() : Array.from(v.keys());
    a = Array.isArray(ks) ? ks.slice() : Array.from(ks);
  } else a = Array.from(__py$iterable(v));
  a.__pytup = 1;
  return a;
}

function set(v) {
  if (v === undefined) return new $Set();
  return new $Set(v);
}

function frozenset(v) {
  if (v === undefined) return new $Set();
  return new $Set(v);
}

class $PyType {
  constructor(n) {
    this.$n = n;
    this.__name__ = n;
  }
  toString() {
    return "<class '" + this.$n + "'>";
  }
}

function type(v) {
  if (v === null || v === undefined) return new $PyType("NoneType");
  if (typeof v === "boolean") return new $PyType("bool");
  if (typeof v === "bigint") return new $PyType("int");
  if (typeof v === "number") {
    return new $PyType(Number.isInteger(v) ? "int" : "float");
  }
  if (typeof v === "string") return new $PyType("str");
  if (Array.isArray(v)) {
    return new $PyType(v.__pytup !== undefined ? "tuple" : "list");
  }
  if (v instanceof Map) {
    return new $PyType(v instanceof $Counter ? "Counter"
      : (v instanceof $DefaultDict ? "defaultdict"
        : (v instanceof $OrderedDict ? "OrderedDict" : "dict")));
  }
  if (v instanceof Set) return new $PyType("set");
  if (v instanceof $Range) return new $PyType("range");
  if (v instanceof $Deque) return new $PyType("deque");
  if (v instanceof $PyType) return new $PyType("type");
  if (typeof v === "function") {
    return new $PyType(v.$name ? "function" : "builtin_function_or_method");
  }
  return new $PyType(typename(v));
}

function dict(v) {
  if (v === undefined) return new $Dict();
  if (v === null) {
    throw pyerr("TypeError", "'NoneType' object is not iterable");
  }
  var r = new $Dict();
  if (v instanceof Map) {
    var ks = typeof v.keys === "function" && v.keys !== Map.prototype.keys
      ? v.keys() : Array.from(v.keys());
    if (!Array.isArray(ks)) ks = Array.from(ks);
    for (var i0 = 0; i0 < ks.length; i0++) r.set(ks[i0], v.get(ks[i0]));
    return r;
  }
  var pairs = __py$listItems(v);
  for (var i = 0; i < pairs.length; i++) {
    var p = pairs[i];
    if (Array.isArray(p) && p.length === 2) {
      r.set(p[0], p[1]);
    } else {
      throw pyerr("TypeError", "dictionary update sequence element " + i +
        " has length " + (Array.isArray(p) ? p.length : 1) +
        "; 2 is required");
    }
  }
  return r;
}

function sorted(a) {
  var rest = Array.prototype.slice.call(arguments, 1);
  var pk = __py$popKw(rest);
  rest = pk[0];
  var kwv = pk[1];
  var key = kwv && kwv.key !== undefined ? kwv.key : null;
  if (key === null && rest.length > 0) key = rest[0];
  var rev = kwv && kwv.reverse ? true : false;
  var src = Array.from(__py$iterable(a));
  var pairs = new Array(src.length);
  for (var i = 0; i < src.length; i++) {
    pairs[i] = [key ? key(src[i]) : src[i], src[i], i];
  }
  pairs.sort(function(p, q) {
    var r = cmp(p[0], q[0]);
    if (r !== 0) return rev ? -r : r;
    return p[2] - q[2];
  });
  var outp = new Array(pairs.length);
  for (var j = 0; j < pairs.length; j++) outp[j] = pairs[j][1];
  return outp;
}

function __py$minmax(args, dir) {
  var rest = Array.prototype.slice.call(args);
  var pk = __py$popKw(rest);
  rest = pk[0];
  var kwv = pk[1];
  var key = kwv && kwv.key !== undefined ? kwv.key : null;
  var items;
  if (rest.length === 0) {
    throw pyerr("TypeError", dir < 0 ? "min() missing 1 required " +
      "positional argument: 'iterable'" : "max() arg is an empty sequence");
  }
  if (rest.length === 1) {
    items = Array.from(__py$iterable(rest[0]));
    if (items.length === 0) {
      if (kwv && "default" in kwv) return kwv.default;
      throw pyerr("ValueError", dir < 0
        ? "min() arg is an empty sequence" : "max() arg is an empty sequence");
    }
  } else {
    items = rest;
  }
  var bestK = key ? key(items[0]) : items[0];
  var best = items[0];
  for (var i = 1; i < items.length; i++) {
    var k = key ? key(items[i]) : items[i];
    var r = cmp(k, bestK);
    if (dir < 0 ? r < 0 : r > 0) {
      bestK = k;
      best = items[i];
    }
  }
  return best;
}

function max() { return __py$minmax(arguments, 1); }
function min() { return __py$minmax(arguments, -1); }

function sum(it, start) {
  var s = start === undefined ? 0 : start;
  if (typeof s === "string") {
    throw pyerr("TypeError", "sum() can't sum strings [use str.join]");
  }
  for (var x of __py$iterable(it)) {
    if (typeof x === "string") {
      throw pyerr("TypeError", "unsupported operand type(s) for +: '" +
        typename(s) + "' and 'str'");
    }
    s = pyadd(s, x);
  }
  return s;
}

function any(it) {
  for (var x of __py$iterable(it)) if (truthy(x)) return true;
  return false;
}

function all(it) {
  for (var x of __py$iterable(it)) if (!truthy(x)) return false;
  return true;
}

function round(v, n) {
  if (typeof v === "bigint") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "string" || v === null || v === undefined) {
    throw pyerr("TypeError", "'" + typename(v) +
      "' object cannot be interpreted as an integer");
  }
  if (typeof v !== "number") {
    throw pyerr("TypeError", "'" + typename(v) +
      "' object cannot be interpreted as an integer");
  }
  if (n === undefined || n === null) {
    if (Number.isInteger(v)) return v;
    return parseFloat(fmt(v, ".0f"));
  }
  if (typeof n === "bigint") n = Number(n);
  if (typeof n === "boolean") n = n ? 1 : 0;
  if (typeof n !== "number" || !Number.isInteger(n)) {
    throw pyerr("TypeError", "'" + typename(n) +
      "' object cannot be interpreted as an integer");
  }
  if (Number.isInteger(v)) return v;
  if (n >= 0) return parseFloat(fmt(v, "." + n + "f"));
  var scale = Math.pow(10, -n);
  return parseFloat(fmt(v / scale, ".0f")) * scale;
}

function pow(a, b, m) {
  if (m !== undefined && m !== null) {
    if (typeof a === "number" && !Number.isInteger(a)) {
      throw pyerr("TypeError", "unsupported operand type(s) for pow(): '" +
        typename(a) + "', '" + typename(b) + "', '" + typename(m) + "'");
    }
    if (typeof b === "number" && !Number.isInteger(b)) {
      throw pyerr("TypeError", "unsupported operand type(s) for pow(): '" +
        typename(a) + "', '" + typename(b) + "', '" + typename(m) + "'");
    }
    if (typeof m === "number" && !Number.isInteger(m)) {
      throw pyerr("TypeError", "unsupported operand type(s) for pow(): '" +
        typename(a) + "', '" + typename(b) + "', '" + typename(m) + "'");
    }
    if (typeof m === "number" && m === 0) {
      throw pyerr("ValueError", "pow() 3rd argument cannot be 0");
    }
    if (typeof m === "bigint" && m === 0n) {
      throw pyerr("ValueError", "pow() 3rd argument cannot be 0");
    }
    var mm = typeof m === "bigint" ? m : BigInt(m);
    if (mm < 0n) {
      throw pyerr("ValueError", "pow() 3rd argument cannot be negative");
    }
    var bb = typeof b === "bigint" ? b : BigInt(b);
    if (bb < 0n) {
      throw pyerr("ValueError", "pow() 2nd argument cannot be negative " +
        "when 3rd argument specified");
    }
    var aa = typeof a === "bigint" ? a : BigInt(a);
    var base = aa % mm;
    if (base < 0n) base += mm;
    var res = 1n;
    while (bb > 0n) {
      if (bb & 1n) res = (res * base) % mm;
      base = (base * base) % mm;
      bb >>= 1n;
    }
    return __py$n(res);
  }
  return pypow(a, b);
}

function divmod(a, b) {
  if (typeof a === "number" && !Number.isInteger(a)) {
    if (b === 0) {
      throw pyerr("ZeroDivisionError", "float division by zero");
    }
    var qf = Math.floor(a / b);
    return tup(qf, a - qf * b);
  }
  if (typeof b === "number" && !Number.isInteger(b)) {
    if (b === 0) {
      throw pyerr("ZeroDivisionError", "float division by zero");
    }
    var qf2 = Math.floor(a / b);
    return tup(qf2, a - qf2 * b);
  }
  var bb2 = typeof b === "bigint" ? b : BigInt(b);
  if (bb2 === 0n) {
    throw pyerr("ZeroDivisionError", "integer division or modulo by zero");
  }
  var aa2 = typeof a === "bigint" ? a : BigInt(a);
  var q = aa2 / bb2;
  var r = aa2 % bb2;
  if (r !== 0n && (r < 0n) !== (bb2 < 0n)) {
    q -= 1n;
    r += bb2;
  }
  return tup(__py$n(q), __py$n(r));
}

function __py$baseOf(v, radix, prefix) {
  var neg = false;
  if (typeof v === "boolean") v = v ? 1 : 0;
  if (typeof v === "number") {
    if (!Number.isInteger(v)) {
      throw pyerr("TypeError",
        "'float' object cannot be interpreted as an integer");
    }
    neg = v < 0;
    v = BigInt(Math.abs(v));
  } else if (typeof v === "bigint") {
    neg = v < 0n;
    v = v < 0n ? -v : v;
  } else {
    throw pyerr("TypeError", "'" + typename(v) +
      "' object cannot be interpreted as an integer");
  }
  return (neg ? "-" : "") + prefix + v.toString(radix);
}

function bin(v) { return __py$baseOf(v, 2, "0b"); }
function hex(v) { return __py$baseOf(v, 16, "0x"); }
function oct(v) { return __py$baseOf(v, 8, "0o"); }

function chr(o) {
  if (typeof o === "boolean") o = o ? 1 : 0;
  if (typeof o === "bigint") o = Number(o);
  if (typeof o !== "number" || !Number.isInteger(o)) {
    throw pyerr("TypeError", "an integer is required (got type " +
      typename(o) + ")");
  }
  if (o < 0 || o > 0x10ffff) {
    throw pyerr("ValueError", "chr() arg not in range(0x110000)");
  }
  return String.fromCodePoint(o);
}

function ord(c) {
  if (typeof c !== "string") {
    throw pyerr("TypeError", "ord() expected string, but " +
      typename(c) + " found");
  }
  if (c.length === 0) {
    throw pyerr("TypeError", "ord() expected a character, but string of " +
      "length 0 found");
  }
  var cp = c.codePointAt(0);
  if (c.length > 1 && !(cp > 0xffff && c.length === 2)) {
    throw pyerr("TypeError", "ord() expected a character, but string of " +
      "length " + c.length + " found");
  }
  return cp;
}

function int(v, base) {
  if (v === undefined) return 0;
  if (v === null) {
    throw pyerr("TypeError", "int() argument must be a string, a " +
      "bytes-like object or a number, not 'NoneType'");
  }
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "bigint") return __py$n(v);
  if (typeof v === "number") {
    if (!Number.isInteger(v)) {
      if (!Number.isFinite(v)) {
        throw pyerr("OverflowError",
          "cannot convert float infinity to integer");
      }
      return Math.trunc(v);
    }
    return v;
  }
  if (typeof v === "string") {
    var st = v.replace(/_/g, "");
    var b = 10;
    if (base !== undefined && base !== null) {
      if (typeof base === "bigint") base = Number(base);
      if (typeof base === "number" && Number.isInteger(base) &&
          (base === 0 || (base >= 2 && base <= 36))) {
        b = base;
      } else {
        throw pyerr("ValueError", "int() base must be >= 2 and <= 36");
      }
    }
    var s2 = st.trim();
    var sign = 1;
    if (s2.charAt(0) === "+") s2 = s2.slice(1);
    else if (s2.charAt(0) === "-") {
      sign = -1;
      s2 = s2.slice(1);
    }
    if (b === 0) {
      if (/^0[xX]/.test(s2)) {
        b = 16;
        s2 = s2.slice(2);
      } else if (/^0[bB]/.test(s2)) {
        b = 2;
        s2 = s2.slice(2);
      } else if (/^0[oO]/.test(s2)) {
        b = 8;
        s2 = s2.slice(2);
      } else {
        b = 10;
      }
    } else if (b !== 10) {
      if ((b === 2 && /^0[bB]/.test(s2)) ||
          (b === 8 && /^0[oO]/.test(s2)) ||
          (b === 16 && /^0[xX]/.test(s2))) {
        s2 = s2.slice(2);
      }
    }
    if (s2.length === 0) {
      throw pyerr("ValueError",
        "invalid literal for int() with base " + b + ": " +
        repr(v));
    }
    var acc2 = 0n;
    for (var i4 = 0; i4 < s2.length; i4++) {
      var d = parseInt(s2.charAt(i4), b);
      if (isNaN(d) || d >= b) {
        throw pyerr("ValueError",
          "invalid literal for int() with base " + b + ": " +
          repr(v));
      }
      acc2 = acc2 * BigInt(b) + BigInt(d);
    }
    return __py$n(sign < 0 ? -acc2 : acc2);
  }
  throw pyerr("TypeError", "int() argument must be a string, a " +
    "bytes-like object or a number, not '" + typename(v) + "'");
}

int.from_bytes = function(bytes, byteorder, signed) {
  if (byteorder === undefined || byteorder === null) byteorder = "big";
  if (byteorder !== "big" && byteorder !== "little") {
    throw pyerr("TypeError", "'" + byteorder +
      "' is not a byte order");
  }
  var sgn = signed === true || (signed !== undefined && signed !== null &&
    truthy(signed));
  var items = Array.isArray(bytes) ? bytes
    : Array.from(__py$iterable(bytes));
  var acc = 0n;
  for (var i = items.length - 1; i >= 0; i--) {
    var byte = items[i];
    if (typeof byte === "bigint") byte = Number(byte);
    if (typeof byte !== "number" || !Number.isInteger(byte) ||
        byte < 0 || byte > 255) {
      throw pyerr("ValueError", "byte must be in range(0, 256)");
    }
    acc = (acc << 8n) | BigInt(byte);
  }
  if (byteorder === "big") {
    var rev = 0n;
    for (var j = 0; j < items.length; j++) {
      rev = (rev << 8n) | BigInt(Number(items[j]));
    }
    acc = rev;
  }
  if (sgn && items.length > 0) {
    var top = BigInt(Number(items[byteorder === "big" ? 0
      : items.length - 1]));
    if ((top & 0x80n) !== 0n) acc -= 1n << BigInt(8 * items.length);
  }
  return __py$n(acc);
};
int.bit_length = function(v) { return NUMM.bit_length(v); };
int.to_bytes = function(v, a) { return NUMM.to_bytes(v,
  Array.prototype.slice.call(arguments, 1)); };

function float(v) {
  if (v === undefined) return 0;
  if (v === null) {
    throw pyerr("TypeError", "float() argument must be a string or a " +
      "number, not 'NoneType'");
  }
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "number") return v;
  if (typeof v === "bigint") return Number(v);
  if (typeof v === "string") {
    var s = v.trim().replace(/_/g, "");
    if (s.length === 0) {
      throw pyerr("ValueError",
        "could not convert string to float: " + repr(v));
    }
    var lo = s.toLowerCase();
    if (lo === "inf" || lo === "+inf" || lo === "infinity" ||
        lo === "+infinity") {
      return Infinity;
    }
    if (lo === "-inf" || lo === "-infinity") return -Infinity;
    if (lo === "nan" || lo === "+nan" || lo === "-nan") return NaN;
    if (/^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$/.test(s)) {
      var r = Number(s);
      if (Number.isNaN(r)) {
        throw pyerr("ValueError",
          "could not convert string to float: " + repr(v));
      }
      return r;
    }
    throw pyerr("ValueError",
      "could not convert string to float: " + repr(v));
  }
  throw pyerr("TypeError", "float() argument must be a string or a " +
    "number, not '" + typename(v) + "'");
}

function bool(v) { return truthy(v); }

function abs(v) {
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "bigint") return v < 0n ? -v : v;
  if (typeof v === "number") {
    if (!Number.isFinite(v)) {
      throw pyerr("OverflowError", "absolute value not finite");
    }
    return Math.abs(v);
  }
  throw pyerr("TypeError", "bad operand type for abs(): '" +
    typename(v) + "'");
}

function format(v, spec) {
  if (spec === undefined || spec === null) return str(v);
  if (typeof v === "string" && (spec === "" || spec === undefined)) return v;
  return fmt(v, spec);
}

function hash(v) {
  if (v === null || v === undefined) return 0;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "number") {
    if (Number.isInteger(v)) return v;
    if (Number.isNaN(v)) return 0;
    return __py$hashStr(String(v));
  }
  if (typeof v === "bigint") {
    if (v >= -9007199254740991n && v <= 9007199254740991n) return Number(v);
    return __py$hashStr(String(v));
  }
  if (typeof v === "string") return __py$hashStr(v);
  if (Array.isArray(v)) {
    var h = 1000003;
    for (var i = 0; i < v.length; i++) {
      h = Math.imul(h ^ hash(v[i]), 16777619);
    }
    return h | 0;
  }
  if (v instanceof Map || v instanceof Set) {
    throw pyerr("TypeError",
      "unhashable type: '" + typename(v) + "'");
  }
  if (typeof v === "object") {
    if (v.__pyhid === undefined) {
      __py$hseq++;
      try {
        Object.defineProperty(v, "__pyhid", { value: __py$hseq });
      } catch (e) {
        return __py$hseq;
      }
    }
    return v.__pyhid;
  }
  return 0;
}

var __py$hseq = 0;

function __py$hashStr(s) {
  var h = 2166136261;
  for (var i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function range(a, b, c) {
  if (arguments.length === 1) return new $Range(0, a, 1);
  if (arguments.length === 2) return new $Range(a, b, 1);
  return new $Range(a, b, c);
}

function reversed(v) {
  if (v === null || v === undefined) {
    throw pyerr("TypeError", "'NoneType' object is not reversible");
  }
  var arr;
  if (typeof v === "string") arr = Array.from(v);
  else if (Array.isArray(v)) arr = v.slice();
  else if (v instanceof $Range) arr = Array.from(v);
  else if (v instanceof $Deque) {
    arr = [];
    for (var i = v.length - 1; i >= 0; i--) arr.push(v.get(i));
  } else if (v instanceof Map || v instanceof Set) {
    throw pyerr("TypeError", "'" + typename(v) + "' object is not reversible");
  } else {
    throw pyerr("TypeError", "'" + typename(v) + "' object is not reversible");
  }
  var idx = arr.length;
  var g = {
    $name: "list_reverseiterator",
    next: function() {
      if (idx <= 0) return { done: true, value: undefined };
      idx--;
      return { done: false, value: arr[idx] };
    }
  };
  g[Symbol.iterator] = function() { return g; };
  return g;
}

function enumerate(it, start) {
  var s = start === undefined ? 0 : start;
  if (typeof s === "boolean") s = s ? 1 : 0;
  if (typeof s === "bigint") {
    var g2 = (function*() {
      var i = s;
      for (var x of __py$iterable(it)) {
        var t = [i, x];
        t.__pytup = 1;
        yield t;
        i = pyadd(i, 1);
      }
    })();
    g2.$name = "enumerate";
    return g2;
  }
  if (typeof s !== "number" || !Number.isInteger(s)) {
    throw pyerr("TypeError", "'" + typename(s) +
      "' object cannot be interpreted as an integer");
  }
  var g = (function*() {
    var i = s;
    for (var x of __py$iterable(it)) {
      var t = [i, x];
      t.__pytup = 1;
      yield t;
      i++;
    }
  })();
  g.$name = "enumerate";
  return g;
}

function zip() {
  var its = Array.prototype.slice.call(arguments);
  if (its.length === 0) {
    var e0 = (function*() {})();
    e0.$name = "zip";
    return e0;
  }
  var g = (function*() {
    var its2 = [];
    for (var i = 0; i < its.length; i++) {
      its2.push(__py$iterable(its[i])[Symbol.iterator]());
    }
    while (true) {
      var vals = [];
      for (var j = 0; j < its2.length; j++) {
        var r = its2[j].next();
        if (r.done) return;
        vals.push(r.value);
      }
      var t = tup.apply(null, vals);
      yield t;
    }
  })();
  g.$name = "zip";
  return g;
}

function map(fn) {
  var its = Array.prototype.slice.call(arguments, 1);
  var g = (function*() {
    if (its.length === 0) return;
    if (its.length === 1) {
      for (var x of __py$iterable(its[0])) yield fn(x);
      return;
    }
    var its2 = [];
    for (var i = 0; i < its.length; i++) {
      its2.push(__py$iterable(its[i])[Symbol.iterator]());
    }
    while (true) {
      var vals = [];
      for (var j = 0; j < its2.length; j++) {
        var r = its2[j].next();
        if (r.done) return;
        vals.push(r.value);
      }
      yield fn.apply(null, vals);
    }
  })();
  g.$name = "map";
  return g;
}

function filter(fn, it) {
  var g = (function*() {
    for (var x of __py$iterable(it)) {
      var keep = fn === null || fn === undefined ? truthy(x)
        : truthy(fn(x));
      if (keep) yield x;
    }
  })();
  g.$name = "filter";
  return g;
}

function input(prompt) {
  if (prompt !== undefined && prompt !== null) out(str(prompt));
  var line = __py$stdinLine();
  if (line === null) {
    throw pyerr("EOFError", "EOF when reading a line");
  }
  return line;
}

str.maketrans = function(a, b, c) {
  if (a !== null && a !== undefined && typeof a === "object" &&
      b === undefined) {
    var m0 = new Map();
    if (a instanceof Map) {
      for (var e of a.entries()) {
        var k = e[0];
        var key = typeof k === "string" ? k : String.fromCodePoint(Number(k));
        m0.set(key, e[1] === null || e[1] === undefined
          ? null : String(e[1]));
      }
      return m0;
    }
    if (typeof a === "object") {
      for (var kk of Object.keys(a)) {
        var vv = a[kk];
        m0.set(/^\d+$/.test(kk) ? String.fromCodePoint(Number(kk)) : kk,
          vv === null || vv === undefined ? null : String(vv));
      }
      return m0;
    }
    throw pyerr("TypeError", "maketrans: bad argument type");
  }
  if (typeof a !== "string" || typeof b !== "string") {
    throw pyerr("TypeError", "maketrans() must be given 2 or 3 strings");
  }
  var m = new Map();
  if (a.length !== b.length) {
    throw pyerr("ValueError", "string.maketrans() must have same length");
  }
  for (var i = 0; i < a.length; i++) m.set(a.charAt(i), b.charAt(i));
  if (c !== undefined && c !== null) {
    if (typeof c !== "string") {
      throw pyerr("TypeError", "maketrans() argument 3 must be str");
    }
    for (var j = 0; j < c.length; j++) m.set(c.charAt(j), null);
  }
  return m;
};

// ---------------------------------------------------------------------------
// bitwise operators + isinstance + import plumbing
// ---------------------------------------------------------------------------
function __py$bitify(v, op, other) {
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "bigint") return v;
  if (typeof v === "number") {
    if (!Number.isInteger(v)) {
      throw pyerr("TypeError", "unsupported operand type(s) for " + op +
        ": '" + typename(v) + "' and '" + typename(other) + "'");
    }
    return v;
  }
  throw pyerr("TypeError", "unsupported operand type(s) for " + op +
    ": '" + typename(v) + "' and '" + typename(other) + "'");
}

function and(a, b) {
  if (typeof a === "boolean" && typeof b === "boolean") return a && b;
  if (a instanceof Set && b instanceof Set) return SETM.intersection(a, [b]);
  if (a instanceof Set || b instanceof Set) {
    throw pyerr("TypeError",
      "unsupported operand type(s) for &: '" + typename(a) + "' and '" +
      typename(b) + "'");
  }
  var x = __py$bitify(a, "&", b), y = __py$bitify(b, "&", a);
  return __py$n(BigInt(x) & BigInt(y));
}

function or(a, b) {
  if (typeof a === "boolean" && typeof b === "boolean") return a || b;
  if (a instanceof Set && b instanceof Set) return SETM.union(a, [b]);
  if (a instanceof Set || b instanceof Set) {
    throw pyerr("TypeError",
      "unsupported operand type(s) for |: '" + typename(a) + "' and '" +
      typename(b) + "'");
  }
  var x = __py$bitify(a, "|", b), y = __py$bitify(b, "|", a);
  return __py$n(BigInt(x) | BigInt(y));
}

function xor(a, b) {
  if (typeof a === "boolean" && typeof b === "boolean") return a !== b;
  if (a instanceof Set && b instanceof Set) {
    return SETM.symmetric_difference(a, [b]);
  }
  if (a instanceof Set || b instanceof Set) {
    throw pyerr("TypeError",
      "unsupported operand type(s) for ^: '" + typename(a) + "' and '" +
      typename(b) + "'");
  }
  var x = __py$bitify(a, "^", b), y = __py$bitify(b, "^", a);
  return __py$n(BigInt(x) ^ BigInt(y));
}

function __py$shiftCount(n, op) {
  if (typeof n === "boolean") return n ? 1 : 0;
  if (typeof n === "bigint") return Number(n);
  if (typeof n === "number" && Number.isInteger(n)) return n;
  throw pyerr("TypeError", "shift count must be an integer, not " +
    typename(n));
}

function shl(a, b) {
  var x = __py$bitify(a, "<<", b);
  var c = __py$shiftCount(b, "<<");
  if (c < 0) throw pyerr("ValueError", "negative shift count");
  return __py$n(BigInt(x) << BigInt(c));
}

function shr(a, b) {
  var x = __py$bitify(a, ">>", b);
  var c = __py$shiftCount(b, ">>");
  if (c < 0) throw pyerr("ValueError", "negative shift count");
  return __py$n(BigInt(x) >> BigInt(c));
}

function invert(a) {
  if (typeof a === "boolean") a = a ? 1 : 0;
  if (typeof a === "number" && !Number.isInteger(a)) {
    throw pyerr("TypeError",
      "bad operand type for unary ~: '" + typename(a) + "'");
  }
  if (typeof a !== "number" && typeof a !== "bigint") {
    throw pyerr("TypeError",
      "bad operand type for unary ~: '" + typename(a) + "'");
  }
  return __py$n(-(BigInt(a) + 1n));
}

function __py$isType(v, name, hint) {
  switch (name) {
    case "int":
      if (hint === 1) return false;
      if (typeof v === "boolean") return true;
      if (typeof v === "bigint") return true;
      return typeof v === "number" && Number.isInteger(v);
    case "float":
      if (hint === 1) return typeof v === "number";
      if (typeof v === "number") return !Number.isInteger(v);
      return false;
    case "bool": return typeof v === "boolean";
    case "str": return typeof v === "string";
    case "bytes":
    case "bytearray": return false;
    case "list": return Array.isArray(v) && v.__pytup === undefined;
    case "tuple": return Array.isArray(v) && v.__pytup !== undefined;
    case "dict": return v instanceof Map;
    case "counter": return v instanceof $Counter;
    case "defaultdict": return v instanceof $DefaultDict;
    case "ordereddict": return v instanceof $OrderedDict;
    case "set": return v instanceof Set;
    case "frozenset": return v instanceof Set;
    case "deque": return v instanceof $Deque;
    case "range": return v instanceof $Range;
    case "None":
    case "NoneType": return v === null || v === undefined;
    case "object": return true;
    case "type": return typeof v === "function";
    case "function": return typeof v === "function";
    default:
      if (v !== null && typeof v === "object" &&
          v.constructor && v.constructor.name === name) {
        return true;
      }
      if (typeof v === "object" && v !== null && v.$namedt === name) {
        return true;
      }
      return false;
  }
}

function isinstance(v, desc, hint) {
  if (Array.isArray(desc)) {
    for (var i = 0; i < desc.length; i++) {
      if (__py$isType(v, desc[i], hint)) return true;
    }
    return false;
  }
  return __py$isType(v, desc, hint);
}

function mod(name) {
  throw pyerr("ImportError", "No module named '" + name + "'");
}

function fset() {
  return new $Set(Array.prototype.slice.call(arguments));
}

// ---------------------------------------------------------------------------
// library modules
// ---------------------------------------------------------------------------
function __py$inbufConsume() {
  var b = __py$stdinBuf();
  __py$inpos = b.length;
}

var $sys = {
  stdin: {
    read: function() {
      var s = __py$stdinRest();
      __py$inbufConsume();
      return s;
    },
    readline: function() {
      var b = __py$stdinBuf();
      if (__py$inpos >= b.length) return "";
      var i = b.indexOf(10, __py$inpos);
      var end = i < 0 ? b.length : i + 1;
      var s = b.toString("utf8", __py$inpos, end);
      __py$inpos = end;
      return s;
    },
    lines: null
  },
  stdout: {
    write: function(s) {
      var t = str(s);
      out(t);
      return t.length;
    },
    flush: function() { return undefined; }
  },
  stderr: {
    write: function(s) {
      var t = str(s);
      var b = Buffer.from(t, "utf8");
      var off = 0;
      var fs = require("fs");
      while (off < b.length) off += fs.writeSync(2, b, off, b.length - off);
      return t.length;
    },
    flush: function() { return undefined; }
  },
  argv: ["python"],
  exit: function(code) {
    process.exit(code === undefined || code === null ? 0
      : Number(code));
  },
  setrecursionlimit: function() { return undefined; },
  getrecursionlimit: function() { return 1000; },
  maxsize: 9223372036854775807,
  platform: "linux",
  version: "3.14.0"
};

function __py$intArg(v, fn) {
  if (typeof v === "boolean") v = v ? 1 : 0;
  if (typeof v === "bigint") return v;
  if (typeof v === "number") {
    if (!Number.isInteger(v)) {
      throw pyerr("TypeError", "must be real number, not " + typename(v));
    }
    return v;
  }
  throw pyerr("TypeError", "must be real number, not " + typename(v));
}

var $math = {
  pi: Math.PI,
  e: Math.E,
  inf: Infinity,
  nan: NaN,
  tau: Math.PI * 2,
  sqrt: function(x) {
    if (typeof x === "bigint") x = Number(x);
    if (x < 0) throw pyerr("ValueError", "math domain error");
    return Math.sqrt(x);
  },
  floor: function(x) {
    if (typeof x === "bigint") return x;
    if (!Number.isFinite(x)) {
      throw pyerr("OverflowError", "cannot convert float infinity to " +
        "integer");
    }
    var r = Math.floor(x);
    if (Number.isSafeInteger(r)) return r;
    return BigInt(r);
  },
  ceil: function(x) {
    if (typeof x === "bigint") return x;
    if (!Number.isFinite(x)) {
      throw pyerr("OverflowError", "cannot convert float infinity to " +
        "integer");
    }
    var r = Math.ceil(x);
    if (Number.isSafeInteger(r)) return r;
    return BigInt(r);
  },
  trunc: function(x) {
    if (typeof x === "bigint") return x;
    if (!Number.isFinite(x)) {
      throw pyerr("OverflowError", "cannot convert float infinity to " +
        "integer");
    }
    var r = Math.trunc(x);
    if (Number.isSafeInteger(r)) return r;
    return BigInt(r);
  },
  fabs: function(x) {
    if (typeof x === "bigint") x = Number(x);
    return Math.abs(x);
  },
  factorial: function(n) {
    if (typeof n === "boolean") n = n ? 1 : 0;
    if (typeof n === "number") {
      if (!Number.isInteger(n)) {
        throw pyerr("TypeError", "'float' object cannot be interpreted " +
          "as an integer");
      }
      n = BigInt(n);
    }
    if (typeof n !== "bigint") {
      throw pyerr("TypeError", "'" + typename(n) +
        "' object cannot be interpreted as an integer");
    }
    if (n < 0n) {
      throw pyerr("ValueError", "factorial() not defined for negative " +
        "values");
    }
    var r = 1n;
    for (var i = 2n; i <= n; i++) r *= i;
    return __py$n(r);
  },
  isqrt: function(n) {
    n = __py$intArg(n, "isqrt");
    var b = typeof n === "bigint" ? n : BigInt(n);
    if (b < 0n) {
      throw pyerr("ValueError", "isqrt() argument must not be negative");
    }
    if (b < 2n) return __py$n(b);
    var x = b;
    var y = (x + 1n) / 2n;
    while (y < x) {
      x = y;
      y = (x + b / x) / 2n;
    }
    return __py$n(x);
  },
  gcd: function(a, b) {
    a = __py$intArg(a, "gcd");
    b = __py$intArg(b, "gcd");
    var x = typeof a === "bigint" ? a : BigInt(a);
    var y = typeof b === "bigint" ? b : BigInt(b);
    if (x < 0n) x = -x;
    if (y < 0n) y = -y;
    while (y !== 0n) {
      var t = x % y;
      x = y;
      y = t;
    }
    return __py$n(x);
  },
  lcm: function(a, b) {
    a = __py$intArg(a, "lcm");
    b = __py$intArg(b, "lcm");
    var x = typeof a === "bigint" ? a : BigInt(a);
    var y = typeof b === "bigint" ? b : BigInt(b);
    if (x < 0n) x = -x;
    if (y < 0n) y = -y;
    if (x === 0n || y === 0n) return 0;
    var xx = x, yy = y;
    while (yy !== 0n) {
      var t = xx % yy;
      xx = yy;
      yy = t;
    }
    return __py$n(x / xx * y);
  },
  comb: function(n, k) {
    n = __py$intArg(n, "comb");
    k = __py$intArg(k, "comb");
    var nn = typeof n === "bigint" ? n : BigInt(n);
    var kk = typeof k === "bigint" ? k : BigInt(k);
    if (nn < 0n || kk < 0n) {
      throw pyerr("ValueError", "must not be negative");
    }
    if (kk > nn) return 0;
    if (kk > nn - kk) kk = nn - kk;
    var r = 1n;
    for (var i = 0n; i < kk; i++) {
      r = r * (nn - i) / (i + 1n);
    }
    return __py$n(r);
  },
  perm: function(n, k) {
    n = __py$intArg(n, "perm");
    var nn = typeof n === "bigint" ? n : BigInt(n);
    if (k === undefined || k === null) k = n;
    else k = __py$intArg(k, "perm");
    var kk = typeof k === "bigint" ? k : BigInt(k);
    if (nn < 0n || kk < 0n) {
      throw pyerr("ValueError", "must not be negative");
    }
    if (kk > nn) return 0;
    var r = 1n;
    for (var i = 0n; i < kk; i++) r *= nn - i;
    return __py$n(r);
  },
  hypot: function() {
    var s = 0;
    for (var i = 0; i < arguments.length; i++) {
      var x = arguments[i];
      if (typeof x === "bigint") x = Number(x);
      s += x * x;
    }
    return Math.sqrt(s);
  },
  exp: function(x) {
    if (typeof x === "bigint") x = Number(x);
    if (x > 709.78) throw pyerr("OverflowError", "math range error");
    var r = Math.exp(x);
    if (!Number.isFinite(r)) throw pyerr("OverflowError", "math range error");
    return r;
  },
  log: function(x, base) {
    if (typeof x === "bigint") x = Number(x);
    if (x <= 0) throw pyerr("ValueError", "math domain error");
    if (base === undefined) return Math.log(x);
    if (typeof base === "bigint") base = Number(base);
    if (base <= 0) throw pyerr("ValueError", "math domain error");
    return Math.log(x) / Math.log(base);
  },
  log2: function(x) {
    if (typeof x === "bigint") x = Number(x);
    if (x <= 0) throw pyerr("ValueError", "math domain error");
    return Math.log2(x);
  },
  log10: function(x) {
    if (typeof x === "bigint") x = Number(x);
    if (x <= 0) throw pyerr("ValueError", "math domain error");
    return Math.log10(x);
  },
  sin: function(x) {
    if (typeof x === "bigint") x = Number(x);
    return Math.sin(x);
  },
  cos: function(x) {
    if (typeof x === "bigint") x = Number(x);
    return Math.cos(x);
  },
  tan: function(x) {
    if (typeof x === "bigint") x = Number(x);
    return Math.tan(x);
  },
  asin: function(x) {
    if (typeof x === "bigint") x = Number(x);
    if (x < -1 || x > 1) throw pyerr("ValueError", "math domain error");
    return Math.asin(x);
  },
  acos: function(x) {
    if (typeof x === "bigint") x = Number(x);
    if (x < -1 || x > 1) throw pyerr("ValueError", "math domain error");
    return Math.acos(x);
  },
  atan: function(x) {
    if (typeof x === "bigint") x = Number(x);
    return Math.atan(x);
  },
  atan2: function(y, x) {
    if (typeof y === "bigint") y = Number(y);
    if (typeof x === "bigint") x = Number(x);
    return Math.atan2(y, x);
  },
  degrees: function(x) {
    if (typeof x === "bigint") x = Number(x);
    return x * 180 / Math.PI;
  },
  radians: function(x) {
    if (typeof x === "bigint") x = Number(x);
    return x * Math.PI / 180;
  }
};

function __py$siftUp(a, i) {
  while (i > 0) {
    var p = (i - 1) >> 1;
    if (cmp(a[i], a[p]) < 0) {
      var t = a[i];
      a[i] = a[p];
      a[p] = t;
      i = p;
    } else break;
  }
}

function __py$siftDown(a, i) {
  var n = a.length;
  while (true) {
    var l = i * 2 + 1;
    if (l >= n) break;
    var s = l;
    if (l + 1 < n && cmp(a[l + 1], a[l]) < 0) s = l + 1;
    if (cmp(a[s], a[i]) < 0) {
      var t = a[i];
      a[i] = a[s];
      a[s] = t;
      i = s;
    } else break;
  }
}

var $heapq = {
  heappush: function(a, x) {
    a.push(x);
    __py$siftUp(a, a.length - 1);
    return undefined;
  },
  heappop: function(a) {
    if (a.length === 0) {
      throw pyerr("IndexError", "pop from an empty heap");
    }
    var top = a[0];
    var last = a.pop();
    if (a.length > 0) {
      a[0] = last;
      __py$siftDown(a, 0);
    }
    return top;
  },
  heapify: function(a) {
    var n = a.length;
    for (var i = (n >> 1) - 1; i >= 0; i--) __py$siftDown(a, i);
    return undefined;
  },
  heappushpop: function(a, x) {
    if (a.length === 0 || cmp(x, a[0]) < 0) return x;
    var top = a[0];
    a[0] = x;
    __py$siftDown(a, 0);
    return top;
  },
  nlargest: function(n, it) {
    var s = sorted(it);
    s.sort(function(p, q) {
      var r = cmp(q, p);
      return r !== 0 ? r : 0;
    });
    return s.slice(0, Number(n));
  },
  nsmallest: function(n, it) {
    return sorted(it).slice(0, Number(n));
  }
};

var $bisect = {
  bisect: function(a, x) { return __py$bisect(a, x, 0, a.length, 1); },
  bisect_right: function(a, x) { return __py$bisect(a, x, 0, a.length, 1); },
  bisect_left: function(a, x) { return __py$bisect(a, x, 0, a.length, 0); },
  insort: function(a, x) {
    var i = __py$bisect(a, x, 0, a.length, 1);
    a.splice(i, 0, x);
    return undefined;
  },
  insort_right: function(a, x) {
    var i = __py$bisect(a, x, 0, a.length, 1);
    a.splice(i, 0, x);
    return undefined;
  },
  insort_left: function(a, x) {
    var i = __py$bisect(a, x, 0, a.length, 0);
    a.splice(i, 0, x);
    return undefined;
  }
};

function __py$bisect(a, x, lo, hi, side) {
  while (lo < hi) {
    var mid = (lo + hi) >> 1;
    var c = side === 0 ? cmp(x, a[mid]) : cmp(a[mid], x);
    if (side === 0 ? c <= 0 : c > 0) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

function __py$chain() {
  var outp = [];
  for (var i = 0; i < arguments.length; i++) {
    for (var x of __py$iterable(arguments[i])) outp.push(x);
  }
  return outp;
}

function __py$accumulate(it, func) {
  var res = [];
  var total;
  var first = true;
  for (var x of __py$iterable(it)) {
    if (first) {
      total = x;
      first = false;
    } else {
      total = func ? func(total, x) : pyadd(total, x);
    }
    res.push(total);
  }
  return res;
}

function __py$permutations(it, r) {
  var items = Array.from(__py$iterable(it));
  var n = items.length;
  var rr = r === undefined || r === null ? n : Number(r);
  if (typeof r === "bigint") rr = Number(r);
  if (rr < 0) throw pyerr("ValueError", "r must be non-negative");
  if (rr > n) return [];
  var res = [];
  var idx = items.map(function(_, i) { return i; });
  var used = new Array(n).fill(false);
  var cur = [];
  function rec() {
    if (cur.length === rr) {
      var t = cur.slice();
      t.__pytup = 1;
      res.push(t);
      return;
    }
    for (var i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(items[i]);
      rec();
      cur.pop();
      used[i] = false;
    }
  }
  rec();
  return res;
}

function __py$combinations(it, r) {
  var items = Array.from(__py$iterable(it));
  var n = items.length;
  var rr = r === undefined || r === null ? n : Number(r);
  if (typeof r === "bigint") rr = Number(r);
  if (rr < 0) throw pyerr("ValueError", "r must be non-negative");
  if (rr > n) return [];
  var res = [];
  var cur = [];
  function rec(start) {
    if (cur.length === rr) {
      var t = cur.slice();
      t.__pytup = 1;
      res.push(t);
      return;
    }
    for (var i = start; i < n; i++) {
      cur.push(items[i]);
      rec(i + 1);
      cur.pop();
    }
  }
  rec(0);
  return res;
}

function __py$product() {
  var its = [];
  for (var i = 0; i < arguments.length; i++) {
    its.push(Array.from(__py$iterable(arguments[i])));
  }
  if (its.length === 0) {
    var t0 = [];
    t0.__pytup = 1;
    return [t0];
  }
  for (var z = 0; z < its.length; z++) {
    if (its[z].length === 0) return [];
  }
  var res = [];
  var idx = new Array(its.length).fill(0);
  var done = false;
  while (!done) {
    var vals = [];
    for (var j = 0; j < its.length; j++) vals.push(its[j][idx[j]]);
    var t = tup.apply(null, vals);
    res.push(t);
    for (var k = its.length - 1; k >= 0; k--) {
      idx[k]++;
      if (idx[k] < its[k].length) break;
      idx[k] = 0;
      if (k === 0) done = true;
    }
  }
  return res;
}

function __py$groupby(it, key) {
  var res = [];
  var src = Array.from(__py$iterable(it));
  var i = 0;
  while (i < src.length) {
    var kv = key ? key(src[i]) : src[i];
    var grp = [];
    while (i < src.length) {
      var cv = key ? key(src[i]) : src[i];
      if (!eq(cv, kv)) break;
      grp.push(src[i]);
      i++;
    }
    var t = [kv, grp];
    t.__pytup = 1;
    res.push(t);
  }
  return res;
}

var $itertools = {
  accumulate: __py$accumulate,
  permutations: __py$permutations,
  combinations: __py$combinations,
  product: __py$product,
  chain: __py$chain,
  groupby: __py$groupby
};

function __py$dequeFactory() {
  return new $Deque();
}

function namedtuple(typename_, fields) {
  var names;
  if (typeof fields === "string") {
    names = fields.trim().split(/\s+/);
  } else {
    names = Array.from(__py$iterable(fields)).map(function(x) {
      return String(x);
    });
  }
  var make = function() {
    var o = {
      $namedt: typename_,
      $fields: names.slice()
    };
    for (var i = 0; i < names.length; i++) {
      o[names[i]] = arguments[i];
    }
    o.toString = function() { return __py$namedtRepr(o); };
    o._fields = names.slice();
    o._asdict = function() {
      var d = new $Dict();
      for (var j = 0; j < names.length; j++) d.set(names[j], o[names[j]]);
      return d;
    };
    o._replace = function(changes) {
      var args = names.map(function(n) {
        return changes !== null && changes !== undefined && n in changes
          ? changes[n] : o[n];
      });
      return make.apply(null, args);
    };
    return o;
  };
  make._fields = names.slice();
  make._make = function(seq) {
    var items = Array.from(__py$iterable(seq));
    return make.apply(null, items);
  };
  return make;
}

function __py$namedtRepr(o) {
  var parts = [];
  for (var i = 0; i < o.$fields.length; i++) {
    parts.push(o.$fields[i] + "=" + repr(o[o.$fields[i]]));
  }
  return o.$namedt + "(" + parts.join(", ") + ")";
}

var $collections = {
  Counter: function(items) {
    var c = new $Counter();
    if (items === undefined || items === null) return c;
    if (items instanceof Map) {
      for (var e of items.entries()) c.set(e[0], e[1]);
      return c;
    }
    if (items instanceof $Dict || items instanceof Map) {
      for (var e2 of items.entries()) c.set(e2[0], e2[1]);
      return c;
    }
    for (var x of __py$iterable(items)) c.set(x, pyadd(c.has(x) ? c.get(x) : 0, 1));
    return c;
  },
  deque: function(items, maxlen) {
    return new $Deque(items, maxlen);
  },
  defaultdict: function(fact) {
    return new $DefaultDict(fact);
  },
  OrderedDict: function() {
    return new $OrderedDict();
  },
  namedtuple: namedtuple,
  ChainMap: null
};

// ---------------------------------------------------------------------------
// re module
// ---------------------------------------------------------------------------
function __py$reCompile(pat, flags, wantGlobal) {
  var fsetChars = [];
  function addF(c) {
    if (fsetChars.indexOf(c) < 0) fsetChars.push(c);
  }
  if (wantGlobal) addF("g");
  if (typeof flags === "number" && flags !== 0) {
    if (flags & 2) addF("i");
    if (flags & 8) addF("m");
    if (flags & 16) addF("s");
  }
  var p = String(pat);
  var m0 = /^\(\?([imsx]+)\)/.exec(p);
  if (m0) {
    for (var ci = 0; ci < m0[1].length; ci++) addF(m0[1].charAt(ci));
    p = p.slice(m0[0].length);
  }
  p = p.replace(/\(\?P<([A-Za-z_][A-Za-z0-9_]*)>/g, "(?<$1>");
  p = p.replace(/\\A/g, "^").replace(/\\Z/g, "$");
  try {
    return new RegExp(p, fsetChars.join(""));
  } catch (e) {
    throw pyerr("re.error", "invalid pattern: " + String(pat));
  }
}

function __py$reGroups(p) {
  var n = 0;
  var inClass = false;
  for (var i = 0; i < p.length; i++) {
    var c = p.charAt(i);
    if (c === "\\") {
      i++;
      continue;
    }
    if (inClass) {
      if (c === "]") inClass = false;
      continue;
    }
    if (c === "[") {
      inClass = true;
      continue;
    }
    if (c === "(") {
      if (p.charAt(i + 1) === "?") {
        var nxt = p.charAt(i + 2);
        if (nxt === "<") {
          var c3 = p.charAt(i + 3);
          if (c3 !== "=" && c3 !== "!") n++;
        }
        continue;
      }
      n++;
    }
  }
  return n;
}

function __py$matchObj(m, s, re) {
  var o = {
    string: s,
    re: re,
    $name: "re.Match",
    pos: 0,
    endpos: s.length,
    group: function() {
      if (arguments.length === 0) return m[0];
      var i = arguments[0];
      if (typeof i === "number") {
        if (i === 0) return m[0];
        if (i < 0 || i >= m.length || m[i] === undefined) {
          if (i >= m.length) {
            throw pyerr("error", "no such group");
          }
          return "";
        }
        return m[i];
      }
      if (m.groups && i in m.groups) {
        var v = m.groups[i];
        return v === undefined ? "" : v;
      }
      throw pyerr("error", "no such group");
    },
    groups: function() {
      var a = [];
      for (var i = 1; i < m.length; i++) {
        a.push(m[i] === undefined ? "" : m[i]);
      }
      a.__pytup = 1;
      return a;
    },
    groupdict: function() {
      var d = new $Dict();
      if (m.groups) {
        for (var k in m.groups) {
          var v = m.groups[k];
          d.set(k, v === undefined ? "" : v);
        }
      }
      return d;
    },
    start: function(i) {
      if (i === undefined || i === null || i === 0) return m.index;
      var g = o.group(i);
      if (g === "") return -1;
      var at = m[0].indexOf(g);
      return m.index + (at < 0 ? 0 : at);
    },
    end: function(i) {
      var st = o.start(i);
      if (st < 0) return -1;
      var g = i === undefined || i === null || i === 0 ? m[0] : o.group(i);
      return st + g.length;
    },
    span: function(i) {
      return tup(o.start(i), o.end(i));
    },
    expand: function(tpl) {
      return __py$replStr(String(tpl), m, m.length - 1);
    },
    lastindex: null,
    lastgroup: null
  };
  return o;
}

function __py$replStr(repl, m, ng) {
  var res = "";
  var i = 0;
  function grp(n) {
    if (n < 0 || n >= m.length || m[n] === undefined) return "";
    return m[n];
  }
  while (i < repl.length) {
    var c = repl.charAt(i);
    if (c === "$" && repl.charAt(i + 1) === "$") {
      res += "$";
      i += 2;
      continue;
    }
    if (c === "\\") {
      var n2 = repl.charAt(i + 1);
      if (n2 === "g") {
        var gt = repl.indexOf(">", i + 3);
        if (gt > 0 && repl.charAt(i + 2) === "<") {
          var ref = repl.slice(i + 3, gt);
          if (/^\d+$/.test(ref)) res += grp(Number(ref));
          else if (m.groups && ref in m.groups) {
            var v = m.groups[ref];
            res += v === undefined ? "" : v;
          } else res += "";
          i = gt + 1;
          continue;
        }
        res += "g";
        i += 2;
        continue;
      }
      if (n2 >= "1" && n2 <= "9") {
        var j = i + 1;
        var ds = "";
        while (j < repl.length && repl.charAt(j) >= "0" &&
               repl.charAt(j) <= "9" && ds.length < 2) {
          ds += repl.charAt(j);
          j++;
        }
        var num = Number(ds);
        if (num === 0) {
          res += m[0];
          i = j;
          continue;
        }
        if (num >= m.length) {
          if (ds.length > 1) {
            res += grp(Number(ds.charAt(0)));
            i = i + 1;
            continue;
          }
          res += "";
          i = j;
          continue;
        }
        res += grp(num);
        i = j;
        continue;
      }
      if (n2 === "n") {
        res += "\n";
        i += 2;
        continue;
      }
      if (n2 === "t") {
        res += "\t";
        i += 2;
        continue;
      }
      if (n2 === "r") {
        res += "\r";
        i += 2;
        continue;
      }
      if (n2 === "\\") {
        res += "\\";
        i += 2;
        continue;
      }
      res += n2;
      i += 2;
      continue;
    }
    res += c;
    i++;
  }
  return res;
}

function __py$reKw(args, names) {
  var pk = __py$popKw(Array.prototype.slice.call(args));
  var a = pk[0];
  var kw = pk[1];
  var o = {};
  for (var i = 0; i < names.length; i++) o[names[i]] = a[i];
  if (kw) {
    for (var k in kw) {
      if (k !== "__pykw" && kw[k] !== undefined) o[k] = kw[k];
    }
  }
  return o;
}

var $re = {
  IGNORECASE: 2,
  MULTILINE: 8,
  DOTALL: 16,
  VERBOSE: 32,
  ASCII: 64,
  I: 2,
  M: 8,
  S: 16,
  X: 32,
  A: 64,
  escape: function(s) {
    return String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  },
  findall: function(pat, s, flags) {
    var __o = __py$reKw(arguments, ["pat", "s", "flags"]);
    pat = __o.pat; s = __o.s; flags = __o.flags;
    if (typeof s !== "string") {
      throw pyerr("TypeError", "expected string or bytes-like object");
    }
    var re = __py$reCompile(pat, flags, true);
    var ng = __py$reGroups(re.source);
    var res = [];
    for (var m of s.matchAll(re)) {
      if (ng === 0) {
        res.push(m[0]);
      } else if (ng === 1) {
        res.push(m[1] === undefined ? "" : m[1]);
      } else {
        var t = [];
        for (var i = 1; i <= ng; i++) {
          t.push(m[i] === undefined ? "" : m[i]);
        }
        t.__pytup = 1;
        res.push(t);
      }
    }
    return res;
  },
  finditer: function(pat, s, flags) {
    var __o = __py$reKw(arguments, ["pat", "s", "flags"]);
    pat = __o.pat; s = __o.s; flags = __o.flags;
    if (typeof s !== "string") {
      throw pyerr("TypeError", "expected string or bytes-like object");
    }
    var re = __py$reCompile(pat, flags, true);
    var res = [];
    for (var m of s.matchAll(re)) res.push(__py$matchObj(m, s, re));
    return res;
  },
  match: function(pat, s, flags) {
    var __o = __py$reKw(arguments, ["pat", "s", "flags"]);
    pat = __o.pat; s = __o.s; flags = __o.flags;
    if (typeof s !== "string") {
      throw pyerr("TypeError", "expected string or bytes-like object");
    }
    var re = __py$reCompile(pat, flags, false);
    re.lastIndex = 0;
    var m = re.exec(s);
    if (m === null) return null;
    if (m.index !== 0) return null;
    return __py$matchObj(m, s, re);
  },
  search: function(pat, s, flags) {
    var __o = __py$reKw(arguments, ["pat", "s", "flags"]);
    pat = __o.pat; s = __o.s; flags = __o.flags;
    if (typeof s !== "string") {
      throw pyerr("TypeError", "expected string or bytes-like object");
    }
    var re = __py$reCompile(pat, flags, false);
    var m = re.exec(s);
    if (m === null) return null;
    return __py$matchObj(m, s, re);
  },
  sub: function(pat, repl, s, count, flags) {
    var __o = __py$reKw(arguments, ["pat", "repl", "s", "count", "flags"]);
    pat = __o.pat; repl = __o.repl; s = __o.s;
    count = __o.count; flags = __o.flags;
    if (typeof s !== "string") {
      throw pyerr("TypeError", "expected string or bytes-like object");
    }
    var re = __py$reCompile(pat, flags, true);
    var ng = __py$reGroups(re.source);
    var limit = count === undefined || count === null ? 0 : Number(count);
    if (limit < 0) limit = 0;
    var outp = "";
    var last = 0;
    var made = 0;
    for (var m of s.matchAll(re)) {
      if (limit > 0 && made >= limit) break;
      outp += s.slice(last, m.index);
      outp += (typeof repl === "function")
        ? str(repl(__py$matchObj(m, s, re)))
        : __py$replStr(String(repl), m, ng);
      last = m.index + m[0].length;
      made++;
    }
    outp += s.slice(last);
    return outp;
  },
  split: function(pat, s, maxsplit, flags) {
    var __o = __py$reKw(arguments, ["pat", "s", "maxsplit", "flags"]);
    pat = __o.pat; s = __o.s; maxsplit = __o.maxsplit;
    flags = __o.flags;
    if (typeof s !== "string") {
      throw pyerr("TypeError", "expected string or bytes-like object");
    }
    var re = __py$reCompile(pat, flags, true);
    var ng = __py$reGroups(re.source);
    var limit = maxsplit === undefined || maxsplit === null ? 0
      : Number(maxsplit);
    var res = [];
    var last = 0;
    var made = 0;
    for (var m of s.matchAll(re)) {
      if (limit > 0 && made >= limit) break;
      if (m.index === 0 && m[0].length === 0) continue;
      res.push(s.slice(last, m.index));
      for (var i = 1; i <= ng; i++) {
        res.push(m[i] === undefined ? "" : m[i]);
      }
      last = m.index + m[0].length;
      made++;
    }
    res.push(s.slice(last));
    return res;
  },
  fullmatch: function(pat, s, flags) {
    var __o = __py$reKw(arguments, ["pat", "s", "flags"]);
    pat = __o.pat; s = __o.s; flags = __o.flags;
    if (typeof s !== "string") {
      throw pyerr("TypeError", "expected string or bytes-like object");
    }
    var re = __py$reCompile("(?:" + pat + ")$", flags, false);
    re.lastIndex = 0;
    var m = re.exec(s);
    if (m === null || m[0].length !== s.length) return null;
    return __py$matchObj(m, s, re);
  }
};

// ---------------------------------------------------------------------------
// runtime namespace emitted code refers to
// ---------------------------------------------------------------------------
var $ = {
  and: and,
  aset: aset,
  at: at,
  aug: aug,
  bound: bound,
  callm: callm,
  cmp: cmp,
  concat: concat,
  contains: contains,
  delitem: delitem,
  dget: dget,
  dict: dict,
  eq: eq,
  fset: fset,
  floordiv: floordiv,
  fmt: fmt,
  iadd: iadd,
  inRange: inRange,
  invert: invert,
  isinstance: isinstance,
  iter: iter,
  kw: kw,
  mod: mod,
  or: or,
  pow: pow,
  pyadd: pyadd,
  pyerr: pyerr,
  pymod: pymod,
  pymul: pymul,
  pyprint: pyprint,
  pysub: pysub,
  repeat: repeat,
  repr: repr,
  shl: shl,
  shr: shr,
  slice: slice,
  sliceAssign: sliceAssign,
  str: str,
  strcat: strcat,
  strf: strf,
  strmul: strmul,
  truediv: truediv,
  truthy: truthy,
  tup: tup,
  unpack: unpack,
  xor: xor,
  sys: $sys,
  math: $math,
  re: $re,
  heapq: $heapq,
  itertools: $itertools,
  bisect: $bisect,
  collections: $collections
};
