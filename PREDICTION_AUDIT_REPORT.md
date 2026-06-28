# 🔍 WorldCupAI — Prediction Audit Report

**Generated**: 2026-06-28 21:05:03

---

## Root Cause Analysis

### Bug Identified: `elo_diff = 0.0` for ALL knockout matches

**Problem**: The June 2026 group stage matches had `home_elo = NaN` and `away_elo = NaN`. The `get_team_latest_state()` function was using the ABSOLUTE latest match (June 2026), where Elo data was missing. The fallback code assigned every team `elo = 1500.0` exactly, resulting in `elo_diff = 0` for all matches. With Elo zeroed out, the model had no information about team strength and predicted near-random outcomes.

**Fix**: Two-pass lookup: Elo from **latest match WITH valid Elo data** (Nov/Dec 2025); form features and rank from absolute latest match (June 2026).

| Match | Old elo_diff | Corrected elo_diff |
|---|---|---|
| Brazil vs Japan | 0.0 (BUG) | **+51** |
| France vs Sweden | 0.0 (BUG) | **+266** |
| Germany vs Paraguay | 0.0 (BUG) | **+121** |
| Argentina vs Cape Verde | 0.0 (BUG) | **+503** |
| Belgium vs Senegal | 0.0 (BUG) | **+26** |
| Spain vs Austria | 0.0 (BUG) | **+259** |
| Netherlands vs Morocco | 0.0 (BUG) | **+57** |
| Portugal vs Croatia | 0.0 (BUG) | **+42** |

---

## Complete Round of 32 Feature Vectors

### South Africa vs Canada

| Feature | Home (South Africa) | Away (Canada) |
|---|---|---|
| FIFA Rank | 57.0 | 40.0 |
| Elo Rating | 1640 | 1824 |
| Elo Diff | **-184.8** | — |
| Rank Diff | **+17** | — |
| Attack Rating | 1.071 | 1.500 |
| Form Win Rate 5 | 0.000 | 0.400 |
| H2H Meetings | 1 | — |

### Germany vs Paraguay

| Feature | Home (Germany) | Away (Paraguay) |
|---|---|---|
| FIFA Rank | 13.0 | 62.0 |
| Elo Rating | 1926 | 1804 |
| Elo Diff | **+121.2** | — |
| Rank Diff | **-49** | — |
| Attack Rating | 2.500 | 0.929 |
| Form Win Rate 5 | 1.000 | 0.600 |
| H2H Meetings | 2 | — |

### Netherlands vs Morocco

| Feature | Home (Netherlands) | Away (Morocco) |
|---|---|---|
| FIFA Rank | 7.0 | 14.0 |
| Elo Rating | 1937 | 1880 |
| Elo Diff | **+56.9** | — |
| Rank Diff | **-7** | — |
| Attack Rating | 1.643 | 1.429 |
| Form Win Rate 5 | 0.400 | 0.600 |
| H2H Meetings | 3 | — |

### Brazil vs Japan

| Feature | Home (Brazil) | Away (Japan) |
|---|---|---|
| FIFA Rank | 5.0 | 18.0 |
| Elo Rating | 1957 | 1906 |
| Elo Diff | **+51.2** | — |
| Rank Diff | **-13** | — |
| Attack Rating | 1.643 | 1.071 |
| Form Win Rate 5 | 0.600 | 0.800 |
| H2H Meetings | 14 | — |

### France vs Sweden

| Feature | Home (France) | Away (Sweden) |
|---|---|---|
| FIFA Rank | 2.0 | 29.0 |
| Elo Rating | 2003 | 1737 |
| Elo Diff | **+265.7** | — |
| Rank Diff | **-27** | — |
| Attack Rating | 1.857 | 1.143 |
| Form Win Rate 5 | 0.800 | 0.600 |
| H2H Meetings | 16 | — |

### Ivory Coast vs Norway

| Feature | Home (Ivory Coast) | Away (Norway) |
|---|---|---|
| FIFA Rank | 38.0 | 50.0 |
| Elo Rating | 1688 | 1840 |
| Elo Diff | **-151.4** | — |
| Rank Diff | **-12** | — |
| Attack Rating | 1.429 | 2.429 |
| Form Win Rate 5 | 0.800 | 0.400 |
| H2H Meetings | 0 | — |

### Mexico vs Ecuador

| Feature | Home (Mexico) | Away (Ecuador) |
|---|---|---|
| FIFA Rank | 17.0 | 27.0 |
| Elo Rating | 1852 | 1892 |
| Elo Diff | **-40.5** | — |
| Rank Diff | **-10** | — |
| Attack Rating | 0.643 | 0.857 |
| Form Win Rate 5 | 0.200 | 0.400 |
| H2H Meetings | 26 | — |

### England vs DR Congo

| Feature | Home (England) | Away (DR Congo) |
|---|---|---|
| FIFA Rank | 4.0 | 60.0 |
| Elo Rating | 1969 | 1635 |
| Elo Diff | **+333.8** | — |
| Rank Diff | **-56** | — |
| Attack Rating | 1.786 | 0.857 |
| Form Win Rate 5 | 0.600 | 0.200 |
| H2H Meetings | 0 | — |

### United States vs Bosnia & Herzegovina

| Feature | Home (United States) | Away (Bosnia & Herzegovina) |
|---|---|---|
| FIFA Rank | 16.0 | 75.0 |
| Elo Rating | 1829 | 1616 |
| Elo Diff | **+213.1** | — |
| Rank Diff | **-59** | — |
| Attack Rating | 1.571 | 1.071 |
| Form Win Rate 5 | 0.400 | 0.600 |
| H2H Meetings | 3 | — |

### Belgium vs Senegal

| Feature | Home (Belgium) | Away (Senegal) |
|---|---|---|
| FIFA Rank | 6.0 | 19.0 |
| Elo Rating | 1857 | 1832 |
| Elo Diff | **+25.7** | — |
| Rank Diff | **-13** | — |
| Attack Rating | 0.786 | 1.143 |
| Form Win Rate 5 | 0.000 | 0.400 |
| H2H Meetings | 0 | — |

### Portugal vs Croatia

| Feature | Home (Portugal) | Away (Croatia) |
|---|---|---|
| FIFA Rank | 8.0 | 12.0 |
| Elo Rating | 1949 | 1907 |
| Elo Diff | **+42.3** | — |
| Rank Diff | **-4** | — |
| Attack Rating | 1.857 | 1.429 |
| Form Win Rate 5 | 0.600 | 0.400 |
| H2H Meetings | 10 | — |

### Spain vs Austria

| Feature | Home (Spain) | Away (Austria) |
|---|---|---|
| FIFA Rank | 3.0 | 22.0 |
| Elo Rating | 2092 | 1833 |
| Elo Diff | **+258.9** | — |
| Rank Diff | **-19** | — |
| Attack Rating | 2.286 | 1.857 |
| Form Win Rate 5 | 0.600 | 0.800 |
| H2H Meetings | 11 | — |

### Switzerland vs Algeria

| Feature | Home (Switzerland) | Away (Algeria) |
|---|---|---|
| FIFA Rank | 15.0 | 46.0 |
| Elo Rating | 1880 | 1783 |
| Elo Diff | **+96.8** | — |
| Rank Diff | **-31** | — |
| Attack Rating | 1.286 | 1.429 |
| Form Win Rate 5 | 0.800 | 0.600 |
| H2H Meetings | 2 | — |

### Argentina vs Cape Verde

| Feature | Home (Argentina) | Away (Cape Verde) |
|---|---|---|
| FIFA Rank | 1.0 | 65.0 |
| Elo Rating | 2060 | 1558 |
| Elo Diff | **+502.7** | — |
| Rank Diff | **-64** | — |
| Attack Rating | 1.929 | 0.929 |
| Form Win Rate 5 | 1.000 | 0.400 |
| H2H Meetings | 0 | — |

### Colombia vs Ghana

| Feature | Home (Colombia) | Away (Ghana) |
|---|---|---|
| FIFA Rank | 9.0 | 64.0 |
| Elo Rating | 1962 | 1587 |
| Elo Diff | **+375.5** | — |
| Rank Diff | **-55** | — |
| Attack Rating | 1.571 | 0.786 |
| Form Win Rate 5 | 0.600 | 0.400 |
| H2H Meetings | 0 | — |

### Australia vs Egypt

| Feature | Home (Australia) | Away (Egypt) |
|---|---|---|
| FIFA Rank | 24.0 | 36.0 |
| Elo Rating | 1816 | 1732 |
| Elo Diff | **+83.3** | — |
| Rank Diff | **-12** | — |
| Attack Rating | 1.000 | 0.786 |
| Form Win Rate 5 | 0.600 | 0.200 |
| H2H Meetings | 2 | — |

