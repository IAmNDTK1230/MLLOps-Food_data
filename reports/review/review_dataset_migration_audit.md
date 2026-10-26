# Review dataset migration audit

Source: `data\foody\source\data_2025_2026.csv`; SHA256: `4a5e4cc1d7b1f9eb9b2d392e551545e6c3b9082bbc8dfe01ddf17002aea8e787`.

```json
{
  "rows": 15639,
  "exact_duplicate_rows": 183,
  "duplicate_review_records": 183,
  "deduplicated_rows": 15456,
  "columns": [
    "Review Time",
    "Rating",
    "Comment",
    "RestaurantID",
    "Restaurant Name",
    "Address"
  ],
  "null_counts": {
    "Review Time": 0,
    "Rating": 0,
    "Comment": 0,
    "RestaurantID": 0,
    "Restaurant Name": 0,
    "Address": 0
  },
  "unique_restaurant_ids": 817,
  "unique_restaurant_names": 817,
  "restaurant_id_min": 4,
  "restaurant_id_max": 3637,
  "invalid_restaurant_ids": 0,
  "invalid_dates": 0,
  "invalid_ratings": 0,
  "replacement_character_rows": 2,
  "date_min": "2025-01-01",
  "date_max": "2026-06-30",
  "restaurant_id_multiple_names": 0,
  "restaurant_name_multiple_ids": 0
}
```
