"""Quick connectivity check — run once to verify Neo4j and ChromaDB are reachable."""

import os
import socket
import ssl
import urllib.request
from dotenv import load_dotenv

load_dotenv(".env")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
print("── ChromaDB ──────────────────────────────────────────────")
try:
    resp = urllib.request.urlopen("http://localhost:8000/api/v1/heartbeat", timeout=5)
    print(f"  Status : {resp.status}")
    print(f"  Body   : {resp.read().decode()}")
    print("  Result : ✓ HEALTHY")
except Exception as e:
    print(f"  Result : ✗ FAILED — {e}")

# ── Neo4j AuraDB (TCP+TLS port check) ─────────────────────────────────────────
print("\n── Neo4j AuraDB ──────────────────────────────────────────")
neo4j_uri = os.environ.get("NEO4J_URI", "")
host = neo4j_uri.replace("neo4j+s://", "").replace("bolt://", "").split(":")[0]
port = 7687
print(f"  Host   : {host}:{port}")
try:
    ctx = ssl.create_default_context()
    with socket.create_connection((host, port), timeout=10) as raw:
        with ctx.wrap_socket(raw, server_hostname=host) as s:
            cipher = s.cipher()
            print(f"  TLS    : {cipher[0] if cipher else 'unknown'}")
    print("  Result : ✓ TCP+TLS reachable (Bolt port open)")
except Exception as e:
    print(f"  Result : ✗ FAILED — {e}")

# ── Neo4j Bolt driver ─────────────────────────────────────────────────────────
print("\n── Neo4j Bolt driver ──────────────────────────────────────")
try:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        connection_timeout=15,
        max_transaction_retry_time=10,
    )
    driver.verify_connectivity()
    with driver.session(database=os.environ["NEO4J_DATABASE"]) as s:
        record = s.run("RETURN 'hello' AS msg").single()
        print(f"  Query  : {record['msg']}")
    driver.close()
    print("  Result : ✓ AUTHENTICATED & QUERY OK")
except Exception as e:
    print(f"  Result : ✗ FAILED — {e}")
