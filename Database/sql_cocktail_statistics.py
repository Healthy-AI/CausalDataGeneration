get_antibiotcsevents = """
SELECT
    hadm_id, label, starttime
FROM inputevents_mv im 
INNER JOIN d_items 
ON im.itemid = d_items.itemid
WHERE d_items.category like 'Antibiotics' ORDER BY hadm_id, starttime"""