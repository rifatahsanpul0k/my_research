const https = require('https');
const fs = require('fs');

const TOKEN = 'KGAT_c9ffdf486721effe0ecf8ebbaaabd38d';

function callTool(toolName, args) {
  return new Promise((resolve, reject) => {
    const postData = JSON.stringify({
      jsonrpc: '2.0',
      id: Date.now(),
      method: 'tools/call',
      params: {
        name: toolName,
        arguments: args
      }
    });

    const options = {
      hostname: 'www.kaggle.com',
      path: '/mcp',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${TOKEN}`,
        'Accept': 'application/json, text/event-stream',
        'Content-Length': Buffer.byteLength(postData)
      }
    };

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => {
        data += chunk;
      });
      res.on('end', () => {
        if (res.statusCode !== 200) {
          return reject(new Error(`HTTP ${res.statusCode}: ${data}`));
        }
        // Parse SSE response
        const lines = data.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const parsed = JSON.parse(line.substring(6));
              return resolve(parsed);
            } catch (e) {
              return reject(new Error(`Failed to parse SSE JSON: ${e.message} in ${line}`));
            }
          }
        }
        resolve(data);
      });
    });

    req.on('error', (e) => reject(e));
    req.write(postData);
    req.end();
  });
}

async function main() {
  const toolName = process.argv[2];
  const argFileOrJson = process.argv[3];

  if (!toolName) {
    console.error('Usage: node kaggle_mcp_client.js <toolName> [argFileOrJson]');
    process.exit(1);
  }

  let args = {};
  if (argFileOrJson) {
    if (fs.existsSync(argFileOrJson)) {
      const content = fs.readFileSync(argFileOrJson, 'utf8');
      args = JSON.parse(content);
    } else {
      args = JSON.parse(argFileOrJson);
    }
  }

  // If args has top-level "request" and user passed full payload or bare request
  try {
    const res = await callTool(toolName, args);
    console.log(JSON.stringify(res, null, 2));
  } catch (err) {
    console.error('Tool execution error:', err.message);
    process.exit(1);
  }
}

main();
