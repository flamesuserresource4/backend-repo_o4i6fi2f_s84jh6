import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from statistics import mean, pstdev
from typing import List, Dict, Any, Optional

ERGAST_BASE = "https://ergast.com/api/f1"

app = FastAPI(title="F1 Performance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def current_season() -> int:
    from datetime import datetime
    return datetime.utcnow().year


def ergast_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{ERGAST_BASE}/{path}.json"
    try:
        resp = requests.get(url, params=params or {}, timeout=12)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream data error: {str(e)}")


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def calc_driver_metrics(season: int) -> List[Dict[str, Any]]:
    # Standings
    data = ergast_get(f"{season}/driverStandings")
    standings_lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    if not standings_lists:
        return []
    standings = standings_lists[0].get("DriverStandings", [])

    # Build base driver map
    drivers: Dict[str, Dict[str, Any]] = {}
    for d in standings:
        drv = d.get("Driver", {})
        cons = d.get("Constructors", [{}])[0]
        code = drv.get("code") or drv.get("permanentNumber") or drv.get("driverId")
        drivers[drv.get("driverId")] = {
            "driverId": drv.get("driverId"),
            "code": code,
            "givenName": drv.get("givenName"),
            "familyName": drv.get("familyName"),
            "nationality": drv.get("nationality"),
            "constructor": cons.get("name"),
            "points": float(d.get("points", 0)),
            "wins": parse_int(d.get("wins", 0)),
            "position": parse_int(d.get("position", 0)),
            "results": [],  # list of dicts with fields: round, grid, position, status, points
        }

    # Race results for the season
    results_data = ergast_get(f"{season}/results/1")  # fetch winners to get rounds count (pagination helper)
    races_total = parse_int(results_data.get("MRData", {}).get("total", 0))
    # Better approach: fetch all race results for all positions
    race_res_all = ergast_get(f"{season}/results")
    races = race_res_all.get("MRData", {}).get("RaceTable", {}).get("Races", [])

    for race in races:
        round_no = parse_int(race.get("round", 0))
        for res in race.get("Results", []):
            drv = res.get("Driver", {})
            drv_id = drv.get("driverId")
            if drv_id not in drivers:
                # Sometimes substitute drivers without standings yet
                cons = res.get("Constructor", {})
                drivers[drv_id] = {
                    "driverId": drv_id,
                    "code": drv.get("code") or drv.get("permanentNumber") or drv_id,
                    "givenName": drv.get("givenName"),
                    "familyName": drv.get("familyName"),
                    "nationality": drv.get("nationality"),
                    "constructor": cons.get("name"),
                    "points": 0.0,
                    "wins": 0,
                    "position": 99,
                    "results": [],
                }
            grid = parse_int(res.get("grid", 0))
            pos_text = res.get("positionText")
            finish_pos = parse_int(res.get("position", 0)) if pos_text and pos_text.isdigit() else None
            status = res.get("status")
            points = float(res.get("points", 0))
            drivers[drv_id]["results"].append({
                "round": round_no,
                "grid": grid,
                "position": finish_pos,
                "status": status,
                "points": points,
            })

    # Qualifying (avg grid if available)
    quali_data = ergast_get(f"{season}/qualifying")
    quali_races = quali_data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    for race in quali_races:
        round_no = parse_int(race.get("round", 0))
        for qual in race.get("QualifyingResults", []):
            drv = qual.get("Driver", {})
            drv_id = drv.get("driverId")
            if drv_id in drivers:
                try:
                    qpos = parse_int(qual.get("position", 0))
                except Exception:
                    qpos = None
                # attach quali position to matching result if exists
                # otherwise, log in a separate record
                found = False
                for r in drivers[drv_id]["results"]:
                    if r["round"] == round_no:
                        r["quali"] = qpos
                        found = True
                        break
                if not found:
                    drivers[drv_id]["results"].append({
                        "round": round_no,
                        "grid": None,
                        "position": None,
                        "status": None,
                        "points": 0.0,
                        "quali": qpos,
                    })

    # Compute metrics
    output: List[Dict[str, Any]] = []
    for drv_id, info in drivers.items():
        results = sorted(info["results"], key=lambda r: r["round"])
        finishes = [r["position"] for r in results if r.get("position")]
        grids = [r["grid"] for r in results if r.get("grid")]
        qualis = [r.get("quali") for r in results if r.get("quali")]
        points_by_round = [r["points"] for r in results]
        dnfs = sum(1 for r in results if (r.get("position") is None and isinstance(r.get("status"), str) and "Finished" not in r.get("status")))

        avg_finish = float(round(mean(finishes), 2)) if finishes else None
        avg_grid = float(round(mean(grids), 2)) if grids else None
        avg_quali = float(round(mean(qualis), 2)) if qualis else None
        consistency = float(round(pstdev(finishes), 2)) if len(finishes) > 1 else None
        last5 = points_by_round[-5:] if points_by_round else []
        form_points = float(round(sum(last5), 1)) if last5 else 0.0

        # Composite performance index (0-100 scale rough)
        # Weighting: points (50%), avg finish (25%), quali/grid (15%), form (10%), penalty for DNFs
        max_points = 1.0  # normalized later based on leader
        perf = {
            **{k: v for k, v in info.items() if k not in ["results"]},
            "avg_finish": avg_finish,
            "avg_grid": avg_grid,
            "avg_quali": avg_quali,
            "consistency": consistency,
            "dnfs": dnfs,
            "form_points_5": form_points,
            "rounds": len(results),
            "results": results,
        }
        output.append(perf)

    # Normalize points for performance index using leader points
    leader_pts = max((d["points"] for d in output), default=1.0)
    leader_form = max((d.get("form_points_5", 0.0) for d in output), default=1.0)

    def inv(value: Optional[float], default: float = 0.0) -> float:
        if value is None or value == 0:
            return default
        return 1.0 / value

    for d in output:
        points_score = (d["points"] / leader_pts) * 100 * 0.5
        finish_score = (inv(d.get("avg_finish"), 0.0) * 100) * 0.25  # lower avg is better
        quali_score = (inv(d.get("avg_quali") or d.get("avg_grid"), 0.0) * 100) * 0.15
        form_score = ((d.get("form_points_5", 0.0) / (leader_form or 1.0)) * 100) * 0.1
        dnf_penalty = min(d.get("dnfs", 0) * 2.0, 15.0)
        perf_index = max(0.0, min(100.0, points_score + finish_score + quali_score + form_score - dnf_penalty))
        d["performance_index"] = round(perf_index, 2)

    # Rank by performance index (desc)
    output.sort(key=lambda x: (-x["performance_index"], x.get("position", 99)))
    # attach rank
    for i, d in enumerate(output, start=1):
        d["rank"] = i

    return output


@app.get("/")
def root():
    return {"message": "F1 Performance API running"}


@app.get("/api/season/summary")
def season_summary(season: Optional[int] = None):
    year = season or current_season()
    drivers = calc_driver_metrics(year)
    return {"season": year, "count": len(drivers), "drivers": drivers}


@app.get("/api/races")
def races(season: Optional[int] = None):
    year = season or current_season()
    data = ergast_get(f"{year}/results")
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    # Simplify structure
    simplified = []
    for r in races:
        simplified.append({
            "round": parse_int(r.get("round", 0)),
            "raceName": r.get("raceName"),
            "date": r.get("date"),
            "circuit": r.get("Circuit", {}).get("circuitName"),
            "location": r.get("Circuit", {}).get("Location", {}).get("locality"),
            "country": r.get("Circuit", {}).get("Location", {}).get("country"),
            "results": [
                {
                    "position": parse_int(res.get("position", 0)) if (res.get("positionText") or "").isdigit() else None,
                    "driverId": res.get("Driver", {}).get("driverId"),
                    "driver": f"{res.get('Driver', {}).get('givenName')} {res.get('Driver', {}).get('familyName')}",
                    "constructor": res.get("Constructor", {}).get("name"),
                    "grid": parse_int(res.get("grid", 0)),
                    "status": res.get("status"),
                    "points": float(res.get("points", 0)),
                }
                for res in r.get("Results", [])
            ],
        })
    return {"season": year, "races": simplified}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Used",
    }
    try:
        from database import db
        if db is not None:
            response["database"] = "✅ Available"
    except Exception:
        pass
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
