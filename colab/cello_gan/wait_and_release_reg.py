# Notebook cell: exec(open('/content/regfiles/wait_and_release_reg.py').read())  - holds the session until the pipeline writes DONE or FAILED (max 4 h), then releases the runtime so no more compute units are used.
import os, time
F = '/content/drive/MyDrive/cello_gan_out/reg'; t0 = time.time()
while not (os.path.exists(F + '/DONE') or os.path.exists(F + '/FAILED')) and time.time() - t0 < 4 * 3600: time.sleep(60)
print('pipeline finished:', 'DONE' if os.path.exists(F + '/DONE') else ('FAILED' if os.path.exists(F + '/FAILED') else 'timeout'), time.ctime())
from google.colab import drive, runtime; drive.flush_and_unmount(); time.sleep(10); runtime.unassign()   # flush first or recent Drive writes are lost
