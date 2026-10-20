"""Preview an exported ClipRank report on the local computer only."""
import argparse
from pathlib import Path
from functools import partial
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--port',type=int,default=8765);a=p.parse_args()
 root=a.directory.resolve()
 if not (root/'review.html').is_file():p.error('Export the review report first')
 server=ThreadingHTTPServer(('127.0.0.1',a.port),partial(SimpleHTTPRequestHandler,directory=str(root)))
 print('ClipRank review: http://127.0.0.1:'+str(server.server_port)+'/review.html',flush=True)
 try:server.serve_forever()
 except KeyboardInterrupt:pass
 finally:server.server_close()
if __name__=='__main__':main()
