# main.py
import asyncio
import logging
import os
import time
from time import monotonic
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from fastapi import FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Logging
# ---------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("reddit-top24h")

# ---------------------------
# Config
# ---------------------------
RAW_SUBREDDITS = [
    "algotrading","CrazyIdeas","MachineLearning","selfhosted","tasker","automation",
    "behindthegifs","BetterEveryLoop","brkb","FinancialPlanning","Futurology","HistoryPorn",
    "learnprogramming","loseit","MovieDetails","oddlysatisfying","programming","RealEstateAdvice",
    "realtors","SecurityAnalysis","CrazyIdeas","deeplearning","hacking","machinelearningnews",
    "UnethicalLifeProTips","AbruptChaos","AnimalsBeingBros","atmbusiness","BeAmazed","beermoney",
    "behindthegifs","ChatGPTPro","coding","CreditCards","darknet","dividends","enrolledagent",
    "Entrepreneur","food","HIFW","ifyoulikeblank","InteriorDesign","investing","kodi","LocalLLaMA",
    "MadeMeSmile","Monero","MovieDetails","movies","Moviesinthemaking","nevertellmetheodds","NFT",
    "nonononoyes","nostalgia","OpenAI","options","passive_income","PlayItAgainSam","PS4","Python",
    "reactiongifs","reactjs","RoomPorn","Scams","Shoestring","shopify","singularity",
    "StocksAndTrading","therewasanattempt","ThisYouComebacks","tipofmytongue","Upwork",
    "ValueInvesting","videos","webdev","whitecoatinvestor","yesyesyesyesno",
]

def clean_sub(s: str) -> str:
    s = s.strip()
    # Remove only a literal "r/" prefix, not any 'r' or '/' characters.
    if s.lower().startswith("r/"):
        s = s[2:]
    return s.strip().lower()

DEFAULT_SUBS = sorted(set(clean_sub(s) for s in RAW_SUBREDDITS))

SUBS_ENV = os.getenv("SUBREDDITS", "").strip()
SUBREDDITS = (
    sorted(set(clean_sub(s) for s in SUBS_ENV.split(",") if s.strip()))
    if SUBS_ENV else DEFAULT_SUBS
)

# Descriptive UA is important for Reddit; allow override via env.
REDDIT_UA = os.getenv(
    "REDDIT_USER_AGENT",
    "RedditTop24h/1.0 (+https://example.com; contact: you@example.com)",
)
HEADERS = {
    "User-Agent": REDDIT_UA,
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))  # per IP for API endpoints

# ---------------------------
# Helpers
# ---------------------------
def cutoff_ts() -> float:
    return time.time() - 24 * 60 * 60

def pick_preview(d: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    img = None
    thumb = None
    try:
        images = d.get("preview", {}).get("images", [])
        if images:
            img = images[0].get("source", {}).get("url")
            if img:
                img = img.replace("&amp;", "&")
            resolutions = images[0].get("resolutions") or []
            if resolutions:
                img = resolutions[-1].get("url", img).replace("&amp;", "&")
    except Exception:
        pass
    t = d.get("thumbnail")
    if isinstance(t, str) and t.startswith("http"):
        thumb = t
    return img, thumb

async def fetch_top_for_sub(session: aiohttp.ClientSession, sub: str) -> Optional[Dict[str, Any]]:
    # Prefer api.reddit.com which usually returns JSON without HTML interstitials
    url = f"https://api.reddit.com/r/{sub}/top?t=day&limit=5&raw_json=1"
    backoff = 1.0
    for attempt in range(4):
        try:
            async with session.get(url, headers=HEADERS, allow_redirects=True) as resp:
                if resp.status == 429:
                    retry_after = resp.headers.get("retry-after")
                    sleep_s = float(retry_after) if retry_after else backoff
                    log.warning("429 from Reddit for %s; sleeping %.1fs", sub, sleep_s)
                    await asyncio.sleep(max(1.0, sleep_s))
                    backoff = min(backoff * 2, 8.0)
                    continue

                if 500 <= resp.status < 600:
                    log.warning("5xx from Reddit for %s: %s", sub, resp.status)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                    continue

                if resp.status in (403, 404):
                    log.info("Skipping %s due to status %s", sub, resp.status)
                    return None

                ct = resp.headers.get("content-type", "")
                if "json" in ct.lower():
                    try:
                        data = await resp.json()
                    except Exception:
                        data = None
                else:
                    # Reddit occasionally serves HTML; try permissive parse then give up
                    try:
                        data = await resp.json(content_type=None)
                    except Exception:
                        data = None
                        log.warning("Non-JSON for %s (ct=%s)", sub, ct)

                if data is None:
                    return None

        except aiohttp.ClientResponseError as e:
            if e.status in (403, 404):
                return None
            log.warning("ClientResponseError for %s (status %s): %s", sub, e.status, e)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
            continue
        except aiohttp.ClientError as e:
            log.warning("ClientError for %s: %s", sub, e)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
            continue

        children = data.get("data", {}).get("children", [])
        if not children:
            return None

        ctoff = cutoff_ts()
        candidates: List[Dict[str, Any]] = []
        for ch in children:
            d = ch.get("data", {})
            created = float(d.get("created_utc", 0))
            if created >= ctoff:
                candidates.append(d)
        if not candidates:
            return None

        best = max(candidates, key=lambda r: r.get("score", 0) or 0)
        img, thumb = pick_preview(best)

        return {
            "subreddit": sub,
            "id": best.get("id"),
            "title": best.get("title"),
            "author": best.get("author"),
            "score": best.get("score"),
            "upvote_ratio": best.get("upvote_ratio"),
            "num_comments": best.get("num_comments"),
            "created_utc": best.get("created_utc"),
            "created_iso": datetime.fromtimestamp(best.get("created_utc", 0), tz=timezone.utc).isoformat(),
            "permalink": f"https://www.reddit.com{best.get('permalink')}",
            "url": best.get("url"),
            "over_18": best.get("over_18", False),
            "is_self": best.get("is_self", False),
            "post_hint": best.get("post_hint"),
            "image": img,
            "thumbnail": thumb,
        }
    return None

async def fetch_all(subs: List[str]) -> List[Dict[str, Any]]:
    timeout = aiohttp.ClientTimeout(total=60)
    conn = aiohttp.TCPConnector(limit=16, ssl=False)
    results: List[Dict[str, Any]] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        tasks = [fetch_top_for_sub(session, s) for s in subs]
        for coro in asyncio.as_completed(tasks):
            item = await coro
            if item:
                results.append(item)
    return results

def sort_posts(posts: List[Dict[str, Any]], sort: str) -> List[Dict[str, Any]]:
    if sort == "comments":
        return sorted(posts, key=lambda r: r.get("num_comments", 0) or 0, reverse=True)
    if sort == "new":
        return sorted(posts, key=lambda r: r.get("created_utc", 0) or 0, reverse=True)
    return sorted(posts, key=lambda r: r.get("score", 0) or 0, reverse=True)

# ---------------------------
# App
# ---------------------------
app = FastAPI(title="Reddit Top 24h")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Request timing and structured logs
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = monotonic()
    response: Response = await call_next(request)
    dur_ms = int((monotonic() - start) * 1000)
    log.info(
        "request path=%s status=%s dur_ms=%s ip=%s ua=%s",
        request.url.path,
        getattr(response, "status_code", 0),
        dur_ms,
        request.client.host if request.client else "unknown",
        request.headers.get("user-agent", "-"),
    )
    return response

# Simple IP rate limit
_rate: Dict[str, List[float]] = {}
def check_rate_limit(ip: str, limit: int = RATE_LIMIT_PER_MIN) -> bool:
    now = time.time()
    window_start = now - 60
    arr = _rate.setdefault(ip, [])
    # drop old
    while arr and arr[0] < window_start:
        arr.pop(0)
    if len(arr) >= limit:
        return False
    arr.append(now)
    return True

# Cache and background refresh
_cache_data: Optional[List[Dict[str, Any]]] = None
_cache_at: float = 0.0
_refresh_task: Optional[asyncio.Task] = None

async def refresh_cache():
    global _cache_data, _cache_at, _metrics
    data = await fetch_all(SUBREDDITS)
    _cache_data = sort_posts(data, "score")
    _cache_at = monotonic()
    _metrics["last_refresh_epoch"] = int(time.time())
    log.info("cache refreshed items=%s", len(_cache_data))

@app.on_event("startup")
async def on_startup():
    async def loop():
        while True:
            try:
                await refresh_cache()
            except Exception as e:
                log.exception("cache refresh failed: %s", e)
            await asyncio.sleep(max(60, CACHE_TTL))
    global _refresh_task
    _refresh_task = asyncio.create_task(loop())

@app.on_event("shutdown")
async def on_shutdown():
    global _refresh_task
    if _refresh_task:
        _refresh_task.cancel()
        try:
            await _refresh_task
        except Exception:
            pass

# ---------------------------
# Health and metrics
# ---------------------------
@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

@app.get("/readiness")
def readiness():
    fresh = _cache_data is not None and (monotonic() - _cache_at) < CACHE_TTL * 2
    status = "ready" if fresh else "warming"
    return JSONResponse({"status": status, "cached_items": len(_cache_data or [])})

_metrics = {
    "requests_total": 0,
    "errors_total": 0,
    "last_refresh_epoch": 0,
}
@app.middleware("http")
async def metrics_mw(request: Request, call_next):
    _metrics["requests_total"] += 1
    try:
        resp = await call_next(request)
        return resp
    except Exception:
        _metrics["errors_total"] += 1
        raise

@app.get("/metrics")
def metrics():
    lines = [
        f'reddit_requests_total {_metrics["requests_total"]}',
        f'reddit_errors_total {_metrics["errors_total"]}',
        f'reddit_cache_age_seconds {int(monotonic() - _cache_at) if _cache_at else -1}',
        f'reddit_cached_items {len(_cache_data or [])}',
        f'reddit_last_refresh_epoch {_metrics["last_refresh_epoch"]}',
    ]
    return PlainTextResponse("\n".join(lines))

# ---------------------------
# API routes
# ---------------------------
@app.get("/api/top-posts")
async def top_posts(
    request: Request,
    subs: Optional[str] = Query(None, description="Comma separated subreddits"),
    force: bool = Query(False, description="Bypass cache"),
    sort: str = Query("score", pattern="^(score|comments|new)$"),
    nsfw: bool = Query(False, description="Include NSFW if true"),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    global _cache_data, _cache_at

    ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(ip):
        return JSONResponse({"error": "rate limited"}, status_code=429)

    chosen = SUBREDDITS if not subs else sorted(
        set(clean_sub(s) for s in subs.split(",") if s.strip())
    )

    now = monotonic()
    fresh = _cache_data is not None and (now - _cache_at) < CACHE_TTL

    if force or not fresh:
        data = await fetch_all(chosen)
        data = sort_posts(data, sort)
        if chosen == SUBREDDITS:
            _cache_data = data
            _cache_at = now
    else:
        data = _cache_data if chosen == SUBREDDITS else [p for p in _cache_data or [] if p.get("subreddit") in chosen]
        data = sort_posts(data, sort)

    if not nsfw:
        data = [p for p in data if not p.get("over_18")]

    total = len(data)
    slice_ = data[offset: offset + limit]
    return JSONResponse({"total": total, "items": slice_})

@app.get("/api/subreddits")
async def list_subreddits():
    return JSONResponse({"subs": SUBREDDITS})

# Static at /static, and serve index.html at /
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")
