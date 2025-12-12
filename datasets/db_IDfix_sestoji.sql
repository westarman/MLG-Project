UPDATE sestoji_attr
SET rfaza = rfaza - 1,
    sksmerni = sksmerni - 1;

UPDATE sestoji_attr
SET pomzas = 0
WHERE pomzas_naziv IS NULL;

UPDATE sestoji_attr
SET negovan = 0
WHERE negovanost_naziv LIKE 'ni podatka%';

UPDATE sestoji_attr
SET zasnova = 0
WHERE zasnova_naziv LIKE 'ni podatka%';

UPDATE sestoji_attr
SET sklep = 0
WHERE sklep_naziv LIKE 'ni podatka%';
