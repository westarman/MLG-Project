-- ONLY DO THE -1 ONCE !!!
UPDATE odseki_attr
SET katgozd = katgozd - 1,
ohranjen = ohranjen - 1,
relief = relief - 1,
kamnina = kamnina - 1,
polpokr = polpokr - 1;

UPDATE odseki_attr
SET pozar = 0
WHERE pozar IS NULL