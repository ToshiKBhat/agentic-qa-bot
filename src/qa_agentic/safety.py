import re

READ_ONLY = ("SELECT", "WITH")
# FORBIDDEN = ("INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "EXPLAIN")
FORBIDDEN = ("INSERT", "UPDATE", "DELETE", "CREATE",  "ALTER", "EXPLAIN")
_SQL_VERB = re.compile(r"^\s*(\w+)", re.IGNORECASE | re.DOTALL)

class SqlSafetyError(ValueError):
    pass

def ensure_read_only(sql: str) -> str:
    verb_match = _SQL_VERB.match(sql or "")
    verb = verb_match.group(1).upper() if verb_match else ""
    if verb in FORBIDDEN or verb not in READ_ONLY:
        raise SqlSafetyError(f"Only read-only SQL allowed. Got '{verb}'.")
    # quick denylist check for other statements
    if any(k in sql.upper() for k in FORBIDDEN):
        raise SqlSafetyError("Forbidden statement detected in SQL body.")
    return sql