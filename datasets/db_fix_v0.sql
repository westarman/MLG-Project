SELECT * FROM pregledovalniksestoji
WHERE
etigl < 0 OR -- negativen posek iglavcev (1)
etlst < 0 OR -- negativen posek listavcev (3)
pompov < 0 OR -- negativna povrsina pomladka (0)
pompov > povrsina OR -- pomladka je več ko površine (2910) <-- tu zgleda, da je small bug, vedno je za 0.1[ha] več pomladka
povrsina - pompov < -0.1 OR -- pomladka je več ko površine (313) <-- to je pa zares, ni bug
povrsina < 0 OR -- negativna povrsina sestoja (0)
(sksmerni_naziv LIKE 'ni ukrepanja%' AND etsku > 0) OR -- posek je določen tam ko 'ni ukrepanja' (414)
(sksmerni_naziv LIKE 'ekocelica (brez ukrepov)%' AND etsku > 0) OR -- posek v ekocelici brez ukrepa (10)
etigl > lzigl OR -- poseka je več ko zaloge za iglavce (1276)
etlst > lzlst OR -- poseka je več ko zaloge za listavce (708)
lzsku > 0 AND 
(lzskdv11 + lzskdv21 + 
 lzskdv30 + lzskdv34 + 
 lzskdv39 + lzskdv41 + 
 lzskdv50 + lzskdv60 + 
 lzskdv70  + lzskdv80  
 NOT BETWEEN 95 AND 105) OR -- delež sestave gozda odstopa za več kot 5% (122) <-- nekateri sestoji sploh nimajo napisano delež, majo pa 3k kubikov zaloge 
pompov > 0 AND
(lzskdv11_m + lzskdv21_m +
 lzskdv30_m + lzskdv34_m +
 lzskdv39_m + lzskdv41_m +
 lzskdv50_m + lzskdv60_m +
 lzskdv70_m + lzskdv80_m 
 NOT BETWEEN 90 AND 110) OR -- delež sestave pomladka odstopa za več kot 10% (2753) <-- našu sm sestoje z več kot 300%, kj to sploh pomeni??? °_°
 lzigl < 1 AND
 (lzskdv11 > 0 OR lzskdv21 > 0 OR lzskdv30 OR lzskdv34 > 0 OR lzskdv39 > 0) OR -- imamo delež iglavcev, čeprav za njih ni zaloge (2)
 lzlst < 1 AND 
 (lzskdv41 > 0 OR lzskdv50 > 0 OR lzskdv60 > 0 OR lzskdv70 > 0 OR lzskdv80 > 0) OR -- imamo delež listavcev, čeprav za njih ni zaloge (2)
 pompov = 0 AND 
(lzskdv11_m + lzskdv21_m +
 lzskdv30_m + lzskdv34_m +
 lzskdv39_m + lzskdv41_m +
 lzskdv50_m + lzskdv60_m +
 lzskdv70_m + lzskdv80_m > 0) OR -- površine mladja ni, je pa za njih delež (0)
 pompov > 0 AND 
(lzskdv11_m + lzskdv21_m +
 lzskdv30_m + lzskdv34_m +
 lzskdv39_m + lzskdv41_m +
 lzskdv50_m + lzskdv60_m +
 lzskdv70_m + lzskdv80_m < 1) OR -- imamo površino mladja, za njih pa ni deleža (557)
etsku/povrsina > 400 -- obsežen posek po m^3/ha (748) <-- če je pravilno pol wow °o°, ko Švedi in Kanadčani...

--(sksmerni_naziv LIKE 'nega debeljaka%' AND sklep_naziv LIKE 'rahel%') OR -- NEGA DEBELJAKA in RAHEL sklep (14247) <-- nism čist prepričan da je napaka, njbrž je kak razlog

/* 
PRI ODSEKIH:
SELECT * FROM pregledovalnikodseki_gozdni
WHERE odprtost NOT BETWEEN 0 AND 100 OR
odprt_gurs NOT BETWEEN 0 AND 100 -- odprtost je ni med 0 in 100% (3) <-- 150%,250%,300% odprti odseki???
*/