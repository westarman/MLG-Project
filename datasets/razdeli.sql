CREATE TABLE Enota AS SELECT
gge,
gge_naziv,
ggo 
FROM odseki_attr;

CREATE TABLE Obmocje AS SELECT
ggo,
ggo_naziv
FROM odseki_attr;

CREATE TABLE Revir AS SELECT
revir, 
revir_naziv,
ggo,
ke
FROM odseki_attr;

CREATE TABLE Krajevna_Enota AS SELECT
ke,
ke_naziv,
ggo
FROM odseki_attr;

CREATE TABLE Relief AS SELECT 
odsek, 
relief, 
lega,
nagib,
nvod AS s_nadmorska_visina,
nvdo AS z_nadmorska_visina,
kamnina,
kamnit AS delez_kamnitosti,
skalnat AS delez_skalovitosti,
odprtost,
odprt_gurs
FROM odseki_attr;

CREATE TABLE Tarife AS SELECT 
odsek,
tarifa_sm AS tarifa_smreka, 
tarifa_je AS tarifa_jelka,
tarifa_oi AS tarifa_ostali_iglavci,
tarifa_bu AS tarifa_bukev,
tarifa_hr AS tarifa_hrast,
tarifa_pl AS tarifa_plemeneniti_listavci,
tarifa_tl AS tarifa_trdi_listavci,
tarifa_ml AS tarifa_mehki_listavci
FROM odseki_attr;

CREATE TABLE Odsek AS SELECT 
odsek,
ggo,
gge,
ke,
revir,
povrsina,
rgr_ggo AS gojitveni_razred_ggo,
rgr_gge AS gojitveni_razred_gge,
katgozd AS kategorija_gozd,
ohranjen AS ohranjenost_gozda,
polpokr AS polozaj_pokrajine,
pozar AS pozar_ogrozenost,
intgosp AS intenzivnost_gospodarjenja,
grt1 AS rastiscni_tip,
rk AS rastiscni_koeficient,
carb_tot_t AS vezani_ogljik,
ponor_c AS letni_ponor_ogljika
FROM odseki_attr;

CREATE TABLE Sestoj AS SELECT 
sestoj,
ggo,
odsek,
povrsina,
rfaza AS razvojna_faza,
sksmerni AS gojitvena_smer,
zasnova,
sklep,
negovan AS negovanost,
pompov AS pomladek_ha,
pomzas AS pomladek_zasnova,
lzigl AS zaloga_iglavcev,
lzlst AS zaloga_listavcev,
lzsku AS zaloga_skupno,
etigl AS posek_iglavcev,
etlst AS posek_listavcev,
etsku AS posek_skupno 
FROM sestoji_attr;

CREATE TABLE Sestava_Gozda AS SELECT
sestoj,
lzskdv11 AS smreka11,
lzskdv11_m AS smreka11_m,
lzskdv21 AS jelka21,
lzskdv21_m AS jelka21_m,
lzskdv30 AS bor30,
lzskdv30_m AS bor30_m,
lzskdv34 AS macesen34,
lzskdv34_m AS macesen34_m,
lzskdv39 AS iglavci39,
lzskdv39_m AS iglavci39_m,
lzskdv41 AS  bukev41,
lzskdv41_m AS bukev41_m,
lzskdv50 AS hrast51,
lzskdv50_m AS hrast51_m,
lzskdv60 AS pListavci60,
lzskdv60_m AS pListavci60_m,
lzskdv70 AS tListavci70,
lzskdv70_m AS tListavci70_m,
lzskdv80 AS mListavci80,
lzskdv80_m AS mListavci80_m
FROM sestoji_attr;