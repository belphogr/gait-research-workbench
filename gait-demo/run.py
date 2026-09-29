"""Double-click start entry; serves built frontend and API locally."""
import sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'vendor'))
sys.path.insert(0,str(BASE))
if __name__=='__main__':
    import uvicorn
    if '--open' in sys.argv:
        import threading, time, urllib.request, webbrowser
        def open_when_ready():
            for _ in range(50):
                try:
                    urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=1).close()
                    webbrowser.open('http://127.0.0.1:8000')
                    return
                except OSError:
                    time.sleep(.3)
        threading.Thread(target=open_when_ready,daemon=True).start()
    print('步态研究工作台：http://127.0.0.1:8000')
    uvicorn.run('backend.app:app',host='127.0.0.1',port=8000)
