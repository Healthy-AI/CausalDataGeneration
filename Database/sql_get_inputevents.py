get_inputevents = """WITH first_admission_time AS
(
  SELECT
      p.subject_id, p.dob
      , MIN (a.admittime) AS first_admittime
      , MIN( ROUND( (cast(admittime as date) - cast(dob as date)) / 365.242,2) )
          AS first_admit_age
  FROM patients p
  INNER JOIN admissions a
  ON p.subject_id = a.subject_id
  GROUP BY p.subject_id, p.dob
)
SELECT
    DISTINCT(d_items.label), hadm_id, first_admit_age,
     CASE
        -- all ages > 89 in the database were replaced with 300
        WHEN first_admit_age > 60
            THEN 'old'
        WHEN first_admit_age <= 60
            THEN 'young'
        END AS age_group
FROM first_admission_time
INNER JOIN inputevents_mv im 
ON im.subject_id = first_admission_time.subject_id
INNER JOIN d_items 
ON im.itemid = d_items.itemid
WHERE d_items.category like 'Antibiotics'"""