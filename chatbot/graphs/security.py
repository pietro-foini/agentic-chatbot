import ast
import re

_SQL_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_SQL_COMMENT_LINE_RE = re.compile(r"--.*?$", re.MULTILINE)

_FORBIDDEN_SQL_KEYWORDS = (
    "ALTER",
    "CALL",
    "COMMENT",
    "COPY",
    "CREATE",
    "DELETE",
    "DO",
    "DROP",
    "EXECUTE",
    "GRANT",
    "INSERT",
    "INTO",
    "LISTEN",
    "LOAD",
    "LOCK",
    "MERGE",
    "NOTIFY",
    "REFRESH",
    "REINDEX",
    "RESET",
    "REVOKE",
    "SECURITY",
    "SET",
    "TRUNCATE",
    "UPDATE",
    "VACUUM",
)

_FORBIDDEN_SQL_FUNCTIONS = (
    "dblink",
    "lo_export",
    "lo_import",
    "pg_execute_server_program",
    "pg_ls_dir",
    "pg_read_binary_file",
    "pg_read_file",
    "pg_sleep",
    "pg_write_file",
)

_PLOT_DENIED_NAMES = frozenset(
    {
        "__builtins__",
        "__import__",
        "breakpoint",
        "builtins",
        "classmethod",
        "compile",
        "delattr",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "input",
        "locals",
        "memoryview",
        "open",
        "os",
        "pathlib",
        "print",
        "quit",
        "requests",
        "setattr",
        "socket",
        "staticmethod",
        "subprocess",
        "super",
        "sys",
        "type",
        "vars",
    }
)

_PLOT_DENIED_CALLS = _PLOT_DENIED_NAMES | frozenset({"__build_class__", "execfile"})


def strip_sql_comments(query: str) -> str:
    """Remove SQL line and block comments before validation."""

    query = _SQL_COMMENT_BLOCK_RE.sub(" ", query)
    query = _SQL_COMMENT_LINE_RE.sub(" ", query)
    return re.sub(r"\s+", " ", query).strip()


def ensure_read_only_sql(query: str) -> str:
    """Accept a single read-only SELECT/CTE statement."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("SQL query must be a non-empty string")

    stripped = strip_sql_comments(query)
    if not stripped:
        raise ValueError("SQL query is empty after removing comments")

    if ";" in stripped.rstrip(";"):
        raise ValueError("Multiple SQL statements are not allowed")

    normalized = stripped.rstrip(";").strip()
    head = normalized.split(None, 1)[0].upper()
    if head not in {"SELECT", "WITH"}:
        raise ValueError("Only SELECT / WITH queries are allowed")

    if re.search(r"\bFOR\s+(UPDATE|SHARE|NO\s+KEY\s+UPDATE|KEY\s+SHARE)\b", normalized, flags=re.IGNORECASE):
        raise ValueError("Locking clauses (FOR UPDATE / FOR SHARE) are not allowed")

    keyword_pattern = r"\b(" + "|".join(_FORBIDDEN_SQL_KEYWORDS) + r")\b"
    if re.search(keyword_pattern, normalized, flags=re.IGNORECASE):
        raise ValueError("Query contains a forbidden SQL command")

    function_pattern = r"\b(" + "|".join(re.escape(name) for name in _FORBIDDEN_SQL_FUNCTIONS) + r")\s*\("
    if re.search(function_pattern, normalized, flags=re.IGNORECASE):
        raise ValueError("Query contains a forbidden SQL function")

    return normalized


def wrap_sql_with_limit(query: str, limit: int = 100) -> str:
    """Force a maximum number of returned rows on a validated SELECT."""

    safe_query = ensure_read_only_sql(query)
    safe_limit = max(1, int(limit))
    return f"SELECT * FROM ({safe_query}) AS agent_query LIMIT {safe_limit}"


def validate_plot_code(code: str) -> ast.Module:
    """Parse plot code and reject imports, dunders, and dangerous calls."""

    if not isinstance(code, str) or not code.strip():
        raise ValueError("Plot code must be a non-empty string")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Plot code is not valid Python: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Imports are not allowed in plot code")

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            raise ValueError("Function and class definitions are not allowed in plot code")

        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Access to private attributes is not allowed in plot code")

        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in _PLOT_DENIED_NAMES):
            raise ValueError(f"Name '{node.id}' is not allowed in plot code")

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _PLOT_DENIED_CALLS:
                raise ValueError(f"Call to '{func.id}' is not allowed in plot code")
            if isinstance(func, ast.Attribute) and func.attr in _PLOT_DENIED_CALLS:
                raise ValueError(f"Call to '{func.attr}' is not allowed in plot code")

    return tree
