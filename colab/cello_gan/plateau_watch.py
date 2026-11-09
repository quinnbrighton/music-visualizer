"""Block until the RVQ run plateaus, reaches END_STEP, or stops writing its log; then exit 0. usage: plateau_watch.py RUN.json RUN.log
Plateau: after MIN_STEP, the best log-mel score (mean over 2, 4 and all stages) of the last WIN evaluations is not TOL better than the best before them."""
import json, os, sys, time
J, LOG = sys.argv[1], sys.argv[2]; MIN_STEP, END, WIN, TOL = int(os.environ.get("MIN_STEP", 5000)), int(os.environ.get("END_STEP", 20000)), int(os.environ.get("WIN", 3)), float(os.environ.get("TOL", 0.003))
def score(e):
    ns = sorted(int(k) for k in e if k.isdigit()); keys = [str(n) for n in (2, 4) if n in ns] + [str(ns[-1])]; return sum(e[k]["lm"] for k in keys) / len(keys)
while True:
    try: E = json.load(open(J))["evals"]
    except Exception: E = []
    if E:
        step = E[-1]["step"]; sc = [score(e) for e in E]
        plateau = len(sc) > WIN and step >= MIN_STEP and min(sc[-WIN:]) > min(sc[:-WIN]) * (1 - TOL)
        print(time.strftime("%H:%M"), "watch: step", step, "score %.4f best %.4f" % (sc[-1], min(sc)), "plateau" if plateau else "", flush=True)
        if plateau or step >= END: print("RVQ done:", "plateau" if plateau else "end", "at step", step, flush=True); sys.exit(0)
    if os.path.exists(LOG) and time.time() - os.path.getmtime(LOG) > 1200: print("RVQ log silent for 20 min; continuing with the latest checkpoint", flush=True); sys.exit(0)
    if os.environ.get("ONCE"): sys.exit(3)
    time.sleep(60)
