from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, model_validator
import socket
import json
import re

llmbp = ChatOllama(model="qwen2.5-coder:3b", base_url="http://localhost:11434")

# NEW: Model for the clarification agent
class QueryCheck(BaseModel):
    is_complete: bool = Field(description="True if the query has enough detail to generate code without hallucinating.")
    missing_details: str = Field(description="If incomplete, a concise, friendly question asking for the missing details (e.g., database type, specific libraries, auth method). If complete, leave empty.")



from typing import List
from pydantic import BaseModel, Field, model_validator

class AiGenContext(BaseModel):
    dependencies: List[str] = Field(description="Upstream module IDs (e.g. 'm1') OR external packages (e.g. 'mysql2').")
    language: str = Field(description="The single language/runtime for the entire blueprint, taken directly from the user's query. If the user didn't specify one, default to 'Python' and note this default was used.")
    code_rules: str = Field(description="Specific logic, database schemas, variable names and algorithms the worker must implement.")

class Module(BaseModel):
    module_id: str = Field(description="Unique identifier formatted as m1, m2, m3, etc.")
    name: str = Field(description="Short technical name of the module.")
    prompt: str = Field(description="Full natural-language task description for this module — sent directly to the code-generation worker as its task instruction.")
    inputs: List[str] = Field(description="Explicit variable or object names required from upstream modules or parameters.")
    returns: List[str] = Field(description="Explicit variable or object names produced and returned by this module.")
    export_signature: str = Field(description="The EXACT function name and parameter order/types this module must export, verbatim — e.g. 'async function signupUser(db_pool_object, username, email, password)'. Other modules that call this one, especially the entry point, MUST use this exact name and argument order.")
    ai_gen_context: AiGenContext = Field(description="Execution details and rules for code generation.")

class Blueprint(BaseModel):
    feature_name: str = Field(description="Name of the feature being decomposed.")
    modules: List[Module] = Field(description="Array of atomic modules ordered by execution hierarchy.")

    @model_validator(mode='after')
    def validate_dependencies(self):
        existing_ids = {m.module_id for m in self.modules}
        for m in self.modules:
            for dep in m.ai_gen_context.dependencies:
                if dep.startswith('m') and dep[1:].isdigit() and dep not in existing_ids:
                    raise ValueError(f"Module {m.module_id} references non-existent module dependency: {dep}")
        return self

    @model_validator(mode='after')
    def validate_language_consistency(self):
        languages = {m.ai_gen_context.language for m in self.modules}
        if len(languages) > 1:
            raise ValueError(f"Modules declare inconsistent languages: {languages}. All modules must target the same runtime unless cross-language boundaries are explicitly intended.")
        return self
    
formated_llm=llmbp.with_structured_output(Blueprint)

class State(TypedDict):
    coding_query: str
    is_ready: bool
    missing_details: str
    bp_resp: str
    module_code: str
    merged_code: str
def clarify_query(state: State) -> State:
    """Agentic loop to ensure the user's query has enough detail to prevent hallucination."""
    query = state.get("coding_query", "")
    if not query:
        query = input("\nDescribe the feature you want to build (be as specific as possible): ")
    
    check_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Requirements Analyst Agent. Your job is to ensure a user's coding request has enough specific detail for an AI to generate working code without hallucinating.
Check for:
1. Database or storage type (e.g., PostgreSQL, SQLite, MongoDB, or "in-memory")
2. Authentication method or specific libraries (e.g., bcrypt, JWT, OAuth)
3. Web framework or runtime (e.g., Flask, FastAPI, Node.js/Express, or "CLI script")
4. Any specific environment variables, API keys, or connection details needed.

If the query is missing critical details that would cause an AI to guess, output:
{
  "is_complete": false,
  "missing_details": "A concise, friendly question asking for the missing details."
}

If the query has enough detail to proceed, output:
{
  "is_complete": true,
  "missing_details": ""
}
"""),
        ("user", "User Query: {query}")
    ])
    
    check_chain = check_prompt | llmbp.with_structured_output(QueryCheck)
    
    current_query = query
    attempts = 0
    max_attempts = 3
    
    while attempts < max_attempts:
        attempts += 1
        result = check_chain.invoke({"query": current_query})
        
        if result.is_complete:
            print("[AI] Query is complete and ready for blueprint generation.\n")
            return {"coding_query": current_query, "is_ready": True, "missing_details": ""}
        else:
            print(f"\n[AI Clarification Needed]: {result.missing_details}")
            user_response = input("Your answer (or type 'skip' to proceed with best guesses): ").strip()
            
            if user_response.lower() == 'skip':
                print("[AI] Proceeding with best guesses. Note: Generated code may require manual adjustments.\n")
                return {"coding_query": current_query, "is_ready": True, "missing_details": "User skipped clarification."}
            
            # Append the clarification to the query for the next iteration
            current_query += f"\n\nAdditional Context Provided by User: {user_response}"
            
    # Fallback if it keeps failing
    print("[AI] Max clarification attempts reached. Proceeding with current query.\n")
    return {"coding_query": current_query, "is_ready": True, "missing_details": "Max attempts reached."}

def bpgen(state):
    q = state["coding_query"] 
    actual_prompt=ChatPromptTemplate.from_messages([("system","""You are the Architect Agent decomposing feature requests into strictly independent, atomic code modules for a distributed team.
        
Rules:
- Prompt Rule: `prompt` must be a complete, self-contained task description — the worker who receives it never sees anything else about the feature or the other modules. State the actual behavior, not a one-line summary.
- Interface Rule: inputs and returns must be explicit variable names or objects. Do not worry about internal variables.
- Do Not Over-Fragment Rule: Each module must be a complete, working unit that can be implemented in isolation. Do not split a single logical task into multiple modules unless it is truly independent and reusable.
- Nomenclature Rule: Variable names in `inputs` MUST exactly match the names declared in the `returns` array of upstream modules. If Module A returns "db_pool_object", Module B must accept "db_pool_object" (not "database" or "db_connection").
- Language Rule: Every module's `ai_gen_context.language` MUST be identical across the whole blueprint — modules that depend on each other's return values cannot be written in different languages.
- Isolation Rule: Use `ai_gen_context.code_rules` for concrete schema details (exact table/column names, types, constraints), library-specific usage (which function calls, which parameters), and edge-case/error-path behavior — specific enough that the worker doesn't have to invent any of it. Dependencies MUST include upstream module_ids (e.g., "m1") if their returned variables are used as inputs, plus any external packages.
- Entry Point Rule: The blueprint MUST include exactly one final module whose sole role is to wire every other module into a single runnable program. It owns startup sequencing (importing/reusing shared resources like connection pools exactly once, never recreating them) and defines how the program is actually invoked (HTTP routes, CLI commands, etc.).
- Entry Point Completeness Rule: The entry point's `prompt` and `code_rules` must explicitly name and invoke every other non-infrastructure module's exported function — using its exact `export_signature` — at least once. Do not summarize with "etc" or ellipses, and do not leave any module's functionality unreferenced. If a module exists in the blueprint, the entry point must state exactly where and how it gets called.
Example Output format in Python (Strictly adhere to this JSON structure):
{{
  "feature_name": "User Authentication",
  "modules": [
    {{
      "module_id": "m1",
      "name": "Database Connection",
      "prompt": "Establish a MySQL connection pool so downstream modules can query the users table without opening their own connections.",
      "inputs": [],
      "returns": ["db_pool_object"],
      "export_signature": "db_pool_object = mysql.connector.pooling.MySQLConnectionPool(...)  # module-level object, not a function",
      "ai_gen_context": {{
        "dependencies": ["mysql-connector-python"],
        "language": "Python",
        "code_rules": "Use mysql.connector.pooling.MySQLConnectionPool with pool_name, pool_size=10, host/user/password/database read from os.environ (DB_HOST, DB_USER, DB_PASSWORD, DB_NAME). Expose db_pool_object as a module-level variable. Do not run queries here."
      }}
    }},
    {{
      "module_id": "m2",
      "name": "User Signup",
      "prompt": "Register a new user by validating the submitted details are unique, hashing their password, and inserting them into the users table.",
      "inputs": ["db_pool_object", "username", "email", "password"],
      "returns": ["new_user_id"],
      "export_signature": "def signup_user(db_pool_object, username, email, password):",
      "ai_gen_context": {{
        "dependencies": ["m1", "bcrypt"],
        "language": "Python",
        "code_rules": "Table `users`: id (INT, auto-increment, PK), username (VARCHAR(50), UNIQUE), email (VARCHAR(255), UNIQUE), password_hash (VARCHAR(255)), created_at (DATETIME). Get a connection via db_pool_object.get_connection(). Query 'SELECT id FROM users WHERE username = %s OR email = %s' with parameters — if a row exists, raise ValueError('User already exists'), do not insert. Hash with bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(). Insert with a parameterized query, commit, close the connection. Return cursor.lastrowid as new_user_id."
      }}
    }},
    {{
      "module_id": "m3",
      "name": "User Authentication",
      "prompt": "Authenticate a login attempt by checking the submitted username and password against the users table.",
      "inputs": ["db_pool_object", "username", "password"],
      "returns": ["is_authenticated", "user_id"],
      "export_signature": "def login_user(db_pool_object, username, password):",
      "ai_gen_context": {{
        "dependencies": ["m1", "bcrypt"],
        "language": "Python",
        "code_rules": "Query 'SELECT id, password_hash FROM users WHERE username = %s' with a parameter. If no row, return (False, None) — do not raise. If found, compare with bcrypt.checkpw(password.encode(), password_hash.encode()). Return (True, id) only on a true match; otherwise (False, None). Never return or log password_hash."
      }}
    }},
    {{
      "module_id": "m4",
      "name": "API Server",
      "prompt": "This module is the single entry point and central orchestrator for the entire feature — it is the only module that actually runs as a program, and every other module's functionality becomes reachable exclusively through the routes defined here. It must expose both user-facing actions this feature supports, with no omissions: a signup route that calls signup_user exactly as exported by m2, and a login route that calls login_user exactly as exported by m3. It owns startup sequencing — import db_pool_object from m1 once at boot and pass it into every downstream call that needs it; never create a second connection pool. Every module in this blueprint that exposes a callable function must be invoked from exactly one place here; no module's logic should be dead code.",
      "inputs": ["db_pool_object"],
      "returns": [],
      "export_signature": "This module is the runnable entry point — no exports, runs via if __name__ == '__main__'.",
      "ai_gen_context": {{
        "dependencies": ["m1", "m2", "m3", "flask", "bcrypt"],
        "language": "Python",
        "code_rules": "At startup, import db_pool_object from the m1 module — do not create a new pool. Create a Flask app. Define POST /signup: parse username, email, password from request.get_json(); call signup_user(db_pool_object, username, email, password) exactly as exported by m2; on success return jsonify({{'user_id': new_user_id}}), 201; on ValueError catch it and return jsonify({{'error': str(e)}}), 409. Define POST /login: parse username, password from request.get_json(); call login_user(db_pool_object, username, password) exactly as exported by m3; if is_authenticated is True return jsonify({{'user_id': user_id}}), 200; if False return jsonify({{'error': 'Invalid credentials'}}), 401. Run with if __name__ == '__main__': app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))). Do not implement signup or login logic directly here — only call m2 and m3."
      }}
    }}
  ]
}}
"""),
        ("user", "{query}")])
    chain=actual_prompt|formated_llm
    ai_resp=chain.invoke({"query":q})
    return {"coding_query": q, "bp_resp": ai_resp.model_dump_json()}
def send_to_mesh(state):
    data = json.loads(state["bp_resp"])
    modules_by_id = {module["module_id"]: module for module in data["modules"]}
    data_to_send = json.dumps({"X": "Y", "payloads": modules_by_id}) + "\n"
    host = '127.0.0.1'
    port = 9000

    module_code = {}
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((host, port))
            client_socket.sendall(data_to_send.encode('utf-8'))

            buffer = ""
            while True:
                chunk = client_socket.recv(65536)
                if not chunk:
                    break  # Node closed the connection — we're done

                buffer += chunk.decode('utf-8')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    status = msg.get('status')

                    if status == 'RECEIVED':
                        print(f"[Python] Node.js acknowledged {msg.get('count')} modules.")
                    elif status == 'RESULTS':
                        print("[Python] Received module results from mesh.")
                        for r in msg['results']:
                            module_code[r['module_id']] = r
                    elif status == 'ERROR':
                        print(f"[Python Error] Mesh reported error: {msg.get('message')}")

    except ConnectionRefusedError:
        print(f"[Python Error] Could not connect to Node.js server at {host}:{port}. Is server.js running?")
    except json.JSONDecodeError as e:
        print(f"[Error] Invalid JSON payload: {e}")

    return {"module_code": json.dumps(module_code)}

def extract_code_from_response(raw_response: str) -> str:
    """The worker model was told 'no markdown, JSON only' but doesn't reliably
    obey that — strip ```json fences if present, then try to parse the
    {"code": "..."} structure. Falls back to the raw text if parsing fails,
    since gemma3:1b sometimes emits malformed JSON (as seen with m2 above)."""
    text = raw_response.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        parsed = json.loads(text)
        return parsed.get("code", text)
    except json.JSONDecodeError:
        return text  # hand back what we got — merge step will see it as-is


def mergegen(state):
    module_code = json.loads(state["module_code"])
    blueprint = json.loads(state["bp_resp"])
    modules_by_id = {m["module_id"]: m for m in blueprint["modules"]}

    sections = []
    for module_id, result in module_code.items():
        spec = modules_by_id.get(module_id, {})
        code = extract_code_from_response(result.get("response", ""))
        sections.append(
            f"### Module {module_id} ({spec.get('name', '')})\n"
            f"Inputs: {', '.join(spec.get('inputs', []))}\n"
            f"Returns: {', '.join(spec.get('returns', []))}\n"
            f"Code:\n{code}"
        )
    merge_input = "\n\n".join(sections)
    print("\n" + "=" * 60)
    print("MERGE INPUT PREVIEW")
    print("=" * 60)
    print(merge_input)
    print("=" * 60)

    while True:
        choice = input("Send this to the merge LLM? (y/n): ").strip().lower()
        if choice in ("y", "yes"):
            break
        if choice in ("n", "no"):
            print("[Python] Merge cancelled by user.")
            return {"merged_code": ""}
        print("Please enter y or n.")

    merge_sys_prompt = """You are the Integrator Agent in a distributed code-generation pipeline.
You receive independently generated code modules. Each was written by an isolated
worker agent that only saw its own interface contract (inputs/returns) and never
saw the other modules' actual code.

Your job is to merge them into a single, coherent, working file.

CRITICAL RULES:
1. PRESERVE CONTRACTS: Each module's declared inputs/returns are the source of truth
   for how modules connect. If a module's actual code doesn't match its declared
   contract (wrong variable name, wrong export shape, missing return), fix the
   mismatch minimally — do not change the contract itself.
2. NO REWRITES: Adapt and wire together what was provided. Do not regenerate a
   module's logic from scratch.
3. FLAG, DON'T GUESS: If a module's code is too broken or ambiguous to reconcile
   safely (invalid syntax, missing logic, contradictory assumptions), do NOT
   invent a fix. Insert a `// TODO(merge): <what's wrong>` comment at that exact
   spot instead.
4. ZERO FLUFF: No explanations outside the JSON output. No markdown fences.
5. STRICT JSON OUTPUT — respond ONLY with:
{{
  "merged_code": "// final merged code as a single string",
  "warnings": ["short description of any mismatch fixed or flagged"]
}}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", merge_sys_prompt),
        ("user", "Feature: {feature_name}\n\nModules to merge:\n{modules}")
    ])

    chain = prompt | llmbp  # raw model — free-form code text, not the Blueprint schema
    resp = chain.invoke({
        "feature_name": blueprint.get("feature_name", ""),
        "modules": merge_input
    })

    raw = resp.content.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    try:
        parsed = json.loads(raw)
        merged_code = parsed.get("merged_code", raw)
        warnings = parsed.get("warnings", [])
        if warnings:
            print("[Python] Merge warnings:", warnings)
    except json.JSONDecodeError:
        merged_code = raw  # model broke its own output contract — hand back raw text
    print("\n=== FINAL MERGED CODE OUTPUT ===")
    return {"merged_code": merged_code}
    

builder = StateGraph(State)

# Add the new clarification node
builder.add_node("clarify_query", clarify_query)
builder.add_node("blueprint generation", bpgen)
builder.add_node("send to mesh", send_to_mesh)
builder.add_node("merge", mergegen) 

# Route: START -> clarify -> blueprint -> mesh -> merge -> END
builder.add_edge(START, "clarify_query")
builder.add_edge("clarify_query", "blueprint generation")
builder.add_edge("blueprint generation", "send to mesh")
builder.add_edge("send to mesh", "merge")

graph = builder.compile()

# Invoke with empty state; the clarify_query node will prompt the user
fin = graph.invoke({"coding_query": "", "is_ready": False, "missing_details": "", "bp_resp": "", "module_code": "", "merged_code": ""})
print("\n=== FINAL MERGED CODE OUTPUT ===")
print(fin["merged_code"])
   
