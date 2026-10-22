/**
 * Simple AI Inference Client for Local LAN
 * Target: Laptop B running Ollama
 */

const PORT = '11434';       // Ollama port
const nodes = [
    { host: '127.0.0.1',     model: 'qwen2.5-coder:3b', timeoutMs: 100000 },
    { host: '192.168.29.69', model: 'qwen2.5-coder:7b', timeoutMs: 100000 }
];

const net = require('net');

const P = 9000;
const HOST = '127.0.0.1';

const server = net.createServer((socket) => {
    let buffer = '';

    socket.on('data', (chunk) => {
        buffer += chunk.toString();

        if (buffer.endsWith('\n')) {
            try {
                const message = JSON.parse(buffer.trim());
                if (message.X === 'Y') {
                    const modulesArray = Object.values(message.payloads);

                    console.log(`[Node.js] Received ${modulesArray.length} modules from Python:`);
                    console.log(JSON.stringify(modulesArray, null, 2));

                    // Immediate ack — framed with \n like everything else on this socket
                    socket.write(JSON.stringify({ status: 'RECEIVED', count: modulesArray.length }) + '\n');

                    // Keep the socket open — results arrive later, asynchronously
                    scheduleModules(modulesArray)
                        .then(results => {
                            console.log('[Node.js] All modules processed:');
                            console.log(JSON.stringify(results, null, 2));

                            const payload = JSON.stringify({ status: 'RESULTS', results }) + '\n';
                            socket.write(payload, (err) => {
                                if (err) console.error('[Node.js] Failed writing results:', err);
                                socket.end();   // done — safe to close now
                            });
                        })
                        .catch(err => {
                            console.error('[Node.js] Scheduling error:', err);
                            socket.write(JSON.stringify({ status: 'ERROR', message: err.message }) + '\n');
                            socket.end();
                        });
                }
            } catch (err) {
                console.error('[Node.js Error] Parsing failed:', err.message);
                socket.write(JSON.stringify({ status: 'ERROR', message: err.message }) + '\n');
                socket.end();
            }
            buffer = '';
        }
    });
});

server.listen(P, HOST, () => {
    console.log(`[Node.js Mesh Server] Listening for Python Blueprints on ${HOST}:${P}`);
});

// Build the actual prompt text sent to the LLM for a given module.
// Customize this to include ai_gen_context.rules, dependencies, etc.
const generateWorkerPrompt = (moduleName, language, inputs, returns, exportSignature, rules, dependencies) => {
    return `You are an autonomous, silent code-generation node in a distributed machine-to-machine mesh network.
Your sole purpose is to write a single, isolated function based on exact specifications.

CRITICAL RULES:
1. ZERO FLUFF: Do not explain your code. Do not add comments like "Here is the code".
2. ABSOLUTELY NO MARKDOWN: Do not wrap your response in ANY backticks.
3. INTERFACE COMPLIANCE: Your function or variable MUST match "Required Export Signature" below verbatim — same name, same parameter order. Inputs and returns must match exactly.
4. STRICT JSON OUTPUT: You must respond ONLY with a valid JSON object.
5. Your "code" value must be complete, real, working ${language} source — never a placeholder, description, or the literal example text shown below.

EXPECTED OUTPUT SHAPE — this shows structure only, replace the value entirely with your actual implementation:
{
  "code": "REPLACE_THIS_WITH_YOUR_ACTUAL_COMPLETE_IMPLEMENTATION"
}

[TASK SPECIFICATIONS]
Module Name: ${moduleName}
Language: ${language}
Required Export Signature: ${exportSignature}
Inputs Required: [${inputs.join(', ')}]
Returns Expected: [${returns.join(', ')}]
Upstream Dependencies: [${dependencies.join(', ')}]

[LOGIC RULES]
${rules}

EXPECTED EXACT OUTPUT FORMAT (Notice there are NO backticks):
{
  "code": "<your code as a single string, using the exact export signature above>"
}`;
};
const acorn = require('acorn'); // npm install acorn

function extractCode(rawResponse) {
    let text = rawResponse.trim();
    const fenceMatch = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    if (fenceMatch) text = fenceMatch[1].trim();
    try {
        const parsed = JSON.parse(text);
        return parsed.code ?? text;
    } catch {
        return text;
    }
}

const { spawnSync } = require('child_process');

function isValidSyntax(code, language) {
    const lang = (language || '').toLowerCase();
    if (lang.includes('python')) {
        return isValidPythonSyntax(code);
    }
    return isValidJsSyntax(code);
}

function isValidJsSyntax(code) {
    try {
        acorn.parse(code, { ecmaVersion: 'latest', sourceType: 'module', allowReturnOutsideFunction: true });
        return true;
    } catch {
        try {
            acorn.parse(code, { ecmaVersion: 'latest', sourceType: 'script', allowReturnOutsideFunction: true });
            return true;
        } catch {
            return false;
        }
    }
}

function isValidPythonSyntax(code) {
    for (const bin of ['python', 'python3']) {
        const result = spawnSync(bin, ['-c', 'import sys, ast; ast.parse(sys.stdin.read())'], {
            input: code,
            encoding: 'utf-8',
            timeout: 5000
        });
        if (!result.error) return result.status === 0;
        // ENOENT means that binary doesn't exist — try the next one
    }
    console.warn('[isValidPythonSyntax] Neither "python" nor "python3" found on PATH — skipping syntax check.');
    return true; // fail open rather than reject everything if Python isn't reachable
}

function looksLikeRealCode(code) {
    const stripped = code.replace(/\/\/.*$/gm, '').replace(/#.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, '').trim();
    if (stripped.length < 20) return false;
    return /(function|=>|const|let|var|class|export|module\.exports|return|await|def |import |elif |self\.)/.test(stripped);
}

async function getInference(node, systemPrompt, userPrompt) {
 const url = `http://${node.host}:${PORT}/api/generate`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), node.timeoutMs);

    const payload = {
        model: node.model,
        system: systemPrompt,
        prompt: userPrompt,
        stream: false,
        options: { temperature: 0.2, num_predict: 1024 }
    };

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: controller.signal
        });
        if (!response.ok) throw new Error(`Ollama Error: ${response.status} ${response.statusText}`);
        const data = await response.json();
        return { ok: true, response: data.response };
    } catch (error) {
        if (error.name === 'AbortError') return { ok: false, error: 'Request timed out' };
        if (error.code === 'ECONNREFUSED') return { ok: false, error: 'Connection refused — is Ollama running on that node?' };
        return { ok: false, error: error.message };
    } finally {
        clearTimeout(timeout);
    }
}


/**
 * Worker-pool scheduler.
 * Spawns one worker per node. Each worker loops: pull next module off the
 * shared queue, run inference, push result, repeat — until the queue is empty.
 * A node that's still busy simply hasn't looped back yet; the moment it
 * finishes it immediately grabs the next queued module. Nodes never idle
 * while work remains, and if there are more nodes than modules the extra
 * workers just find an empty queue and exit right away.
 */
async function scheduleModules(modulesArray) {
    const queue = [...modulesArray];
    const results = [];

    async function worker(node) {
    while (queue.length > 0) {
        const module = queue.shift();
        console.log(`[Node.js] -> ${node} picked up ${module.module_id} (${queue.length} remaining in queue)`);

        const { name, prompt: taskPrompt, inputs, returns, export_signature, ai_gen_context } = module;
const { language, code_rules, dependencies } = ai_gen_context;
const systemPrompt = generateWorkerPrompt(name, language, inputs, returns, export_signature, code_rules, dependencies);

let result, code, attempt = 0;
const maxAttempts = 3;
do {
    attempt++;
    result = await getInference(node, systemPrompt, taskPrompt);
    if (result.ok) {
        code = extractCode(result.response);
        if (!looksLikeRealCode(code)) { result.ok = false; result.error = 'empty_or_placeholder'; }
        else if (!isValidSyntax(code, language)) { result.ok = false; result.error = 'invalid_syntax'; }
    }
} while (!result.ok && attempt < maxAttempts);

        console.log(`[Node.js] <- ${node} finished ${module.module_id} (ok=${result.ok}, attempts=${attempt})`);
        results.push({ module_id: module.module_id, node, attempts: attempt, code: result.ok ? code : null, ...result });
    }
}

    // One worker per node, all launched concurrently, all draining the same queue.
    await Promise.all(nodes.map(node => worker(node)));
    return results;
}