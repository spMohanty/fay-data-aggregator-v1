SELECT
  p.id AS user_id,
  p.cohort,
  g.read_at,
  g.val,
  g.read_at::TIMESTAMPTZ AT TIME ZONE p.timezone AS loc_read_at
FROM glucose_readings g
JOIN faydesc_participants p ON p.id = g.user_id
WHERE
  g.read_at BETWEEN -- TODO, use actual study dates
    COALESCE(p.mfr_started_at, '2018-10-01') AND
    COALESCE(p.mfr_ended_at, 'infinity')
ORDER BY g.user_id, g.read_at
