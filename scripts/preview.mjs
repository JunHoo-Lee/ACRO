import { createServer } from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../docs/', import.meta.url));
const mime = {'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.json':'application/json','.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.mp4':'video/mp4','.pdf':'application/pdf','.zip':'application/zip','.ttf':'font/ttf'};
const server = createServer((request, response) => {
  try {
    const relative = decodeURIComponent(new URL(request.url, 'http://localhost').pathname).replace(/^\/+/, '');
    let path = resolve(root, relative || 'index.html');
    if (!path.startsWith(resolve(root)+sep)) {response.writeHead(403).end();return;}
    if (statSync(path).isDirectory()) path = resolve(path, 'index.html');
    const size = statSync(path).size;
    const headers = {'Content-Type':mime[extname(path)] || 'application/octet-stream','Accept-Ranges':'bytes','Cache-Control':'no-store'};
    const match = /^bytes=(\d+)-(\d*)$/.exec(request.headers.range || '');
    if (match) {
      const start = Number(match[1]);const end = Math.min(match[2] ? Number(match[2]) : size-1,size-1);
      if (start> end || start>=size) {response.writeHead(416,{'Content-Range':`bytes */${size}`}).end();return;}
      response.writeHead(206,{...headers,'Content-Length':end-start+1,'Content-Range':`bytes ${start}-${end}/${size}`});
      if (request.method === 'HEAD') response.end();else createReadStream(path,{start,end}).pipe(response);
    } else {
      response.writeHead(200,{...headers,'Content-Length':size});
      if (request.method === 'HEAD') response.end();else createReadStream(path).pipe(response);
    }
  } catch {response.writeHead(404).end('Not found');}
});
server.listen(4318,'127.0.0.1',() => console.log('ACRO preview: http://127.0.0.1:4318'));
