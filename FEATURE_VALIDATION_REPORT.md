# ✅ WorldCupAI — Feature Validation Report

**Generated**: 2026-06-28 21:05:03

---

## Step 4: Elo Verification

✅ **Fixed**: All teams now use validated Elo from Nov/Dec 2025.

## Step 5: FIFA Ranking Interpretation

✅ Lower rank = stronger team. `rank_diff = home_rank - away_rank`: negative value means home team has better (lower) rank. The model was trained to interpret this correctly.

## Step 6: Home/Away Column Assignment

✅ `home_attack_rating`, `home_form_*` always assigned to the designated home team. No column swapping detected.

## Team-Level Feature Audit

| Team | Elo | FIFA Rank | Attack Rating | Win Rate (5) | Win Rate (10) |
|---|---|---|---|---|---|
| Algeria | 1783 | 46 | 1.429 | 0.600 | 0.700 |
| Argentina | 2060 | 1 | 1.929 | 1.000 | 0.900 |
| Australia | 1816 | 24 | 1.000 | 0.600 | 0.500 |
| Austria | 1833 | 22 | 1.857 | 0.800 | 0.800 |
| Belgium | 1857 | 6 | 0.786 | 0.000 | 0.400 |
| Bosnia & Herzegovina | 1616 | 75 | 1.071 | 0.600 | 0.600 |
| Brazil | 1957 | 5 | 1.643 | 0.600 | 0.500 |
| Canada | 1824 | 40 | 1.500 | 0.400 | 0.400 |
| Cape Verde | 1558 | 65 | 0.929 | 0.400 | 0.400 |
| Colombia | 1962 | 9 | 1.571 | 0.600 | 0.600 |
| Croatia | 1907 | 12 | 1.429 | 0.400 | 0.600 |
| DR Congo | 1635 | 60 | 0.857 | 0.200 | 0.300 |
| Ecuador | 1892 | 27 | 0.857 | 0.400 | 0.400 |
| Egypt | 1732 | 36 | 0.786 | 0.200 | 0.300 |
| England | 1969 | 4 | 1.786 | 0.600 | 0.700 |
| France | 2003 | 2 | 1.857 | 0.800 | 0.800 |
| Germany | 1926 | 13 | 2.500 | 1.000 | 1.000 |
| Ghana | 1587 | 64 | 0.786 | 0.400 | 0.400 |
| Ivory Coast | 1688 | 38 | 1.429 | 0.800 | 0.800 |
| Japan | 1906 | 18 | 1.071 | 0.800 | 0.600 |
| Mexico | 1852 | 17 | 0.643 | 0.200 | 0.300 |
| Morocco | 1880 | 14 | 1.429 | 0.600 | 0.600 |
| Netherlands | 1937 | 7 | 1.643 | 0.400 | 0.600 |
| Norway | 1840 | 50 | 2.429 | 0.400 | 0.600 |
| Paraguay | 1804 | 62 | 0.929 | 0.600 | 0.400 |
| Portugal | 1949 | 8 | 1.857 | 0.600 | 0.800 |
| Senegal | 1832 | 19 | 1.143 | 0.400 | 0.600 |
| South Africa | 1640 | 57 | 1.071 | 0.000 | 0.200 |
| Spain | 2092 | 3 | 2.286 | 0.600 | 0.600 |
| Sweden | 1737 | 29 | 1.143 | 0.600 | 0.300 |
| Switzerland | 1880 | 15 | 1.286 | 0.800 | 0.700 |
| United States | 1829 | 16 | 1.571 | 0.400 | 0.600 |

## NaN Scan Results

Total issues found: **0** (all resolved by Elo fix)

