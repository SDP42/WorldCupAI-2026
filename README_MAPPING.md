# 🗺️ WorldCupAI — Entity Resolution & Mapping Documentation

> Documentation of the entity resolution and name harmonization process for team and country names.

---

## 1. The Challenge of Name Inconsistency

When merging raw data from different sources, team names often vary slightly. For example:
- **USA** in Elo ratings $\rightarrow$ **United States** in results.csv
- **Korea Republic** in FIFA rankings $\rightarrow$ **South Korea** in results.csv
- **Côte d'Ivoire** in FIFA rankings $\rightarrow$ **Cote d'Ivoire** in results.csv
- **Zaire** in historical data $\rightarrow$ **Democratic Republic of the Congo** in results.csv

If we perform standard joins on these columns, the joins will fail, resulting in missing values.

---

## 2. Harmonization Strategy

We established **results.csv** as the canonical name registry. The `TeamHarmonizer` class maps all name variants from other datasets to this canonical list.

The mapping is saved in:
[`mappings/team_name_mapping.csv`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/mappings/team_name_mapping.csv)

### Core Mapping Decisions

| Variant Name | Canonical Name | Rationale |
|---|---|---|
| `USA` | `United States` | Standardize abbreviation |
| `Korea Republic` | `South Korea` | Standardize FIFA naming |
| `IR Iran` | `Iran` | Remove prefix |
| `Côte d'Ivoire` | `Cote d'Ivoire` | Remove accents |
| `Congo DR` | `Democratic Republic of the Congo` | Standardize Congo naming |
| `Zaire` | `Democratic Republic of the Congo` | Map historical country to modern name |
| `USSR` | `Russia` | Map historical country to successor |
| `German DR` | `East Germany` | Keep historical distinction |

---

## 3. Logged Mapping Decisions

During pipeline execution, any team name that does not have an explicit mapping is logged. If the team name is already in the canonical list, it maps to itself. If it's a new variant, the harmonizer logs it as a warning, allowing us to update the mapping table in future iterations.
