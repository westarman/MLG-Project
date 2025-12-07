# ZGS PROJEKT ZA OCENITEV POSEKA IN DRUGIH ATRIBUTOV GOZDA

Naš glavni cilj bo zgraditi RDL model, ki bo na podlagi meta podatkov odsekov za sestoje ter strukture k-sosednih sestojev lahko sam napovedal določene atribute sestojev. Model bo tako lahko prispeval k učinkovitejši in bolj konsistentni pripravi prihodnjih gozdno gospodarskih načrtov.


## PODATKI IZ PREGLEDOVALNIKA

Website: [Pregledovalnik podatkov o gozdovih](https://prostor.zgs.gov.si/pregledovalnik/)

Full raw dataset encompassing the entire country:

- (290mb)[50k_all_odseki.sqlite](https://drive.google.com/file/d/1-k_Y9iBYT9Aj8qa6Ns19J-wyWem4eqIh/view?usp=drive_link) (290mb)
- (673mb)[350k_all_sestoji.sqlite](https://drive.google.com/file/d/1B5DAeCr2gHqvvHo4pdXN9eu8m2iEm7Vg/view?usp=drive_link) (673mb)


Hierarhija enot:  
GGO > Krajevne enote > GGE > Revirji > Odseki > Sestoji

Trenutno stanje podatkov v bazi:
![DB Data](docs/current_db_shema.png)

### Odsek:

| Attribute | Description |
|-----------|-------------|
| ggo | gozdno gospodarkso območje |
| ke | krajevna enota |
| gge | gozdno gospodarksa enota |
| revir | … |
| odsek | … |
| povrsina[ha] | … |
| gojitveni razred ggo | … |
| gojitveni razred gge | … |
| kategorija gozda | … |
| ohranjenost gozda | … |
| polozaj pokrajine | … |
| pozarna ogrozenost | … |
| intenzivnost gospodarjenja | … |
| (vezani ogljik, letni ponor ogljika) | mogoče relevantno mogoče ne? |
| nadm. višina[m] (min,max) | … |
| kamnina | apnenec, dolomit, fliš, diluvialna ilovica, morena karbonatna... |
| delež kamnitosti[%] | delež kamnine, ki je označen zgoraj (preveri) |
| delež skalovitosti[%] | delež skalovja, ki pokriva tla |
| nagib[˚] | … |
| odprtost[%] | odprtost zaradi vlak (vlaka = gozdna "cesta") |
| odprtost za gurs | to baje vključuje odprtost zarad vlak in cest nasplošno (preveri) |
| relief | … |
| lega | nagib glede na kardinalno smer (S, J, V, Z, SZ, JV...) |

### Sestoj:

| Attribute | Description |
|-----------|-------------|
| odsek | … |
| sestoj | … |
| povrsina[ha] | … |
| razvojna faza | … |
| gojitvena smer | … |
| zasnova | … |
| sklep | … |
| negovanost | … |
| površina pomladka[ha] | … |
| pomladek zasnova | … |
| sestava gozda | jelke[%], bukve[%], mehki listavci[%] … |
| zaloga iglavcev [m^3] | trenutna zaloga iglavcev v sestoju |
| zaloga listavcev[m^3] | trenutna zaloga listavcev v sestoju |
| posek iglavcev | naš prediction! |
| posek listavcev | naš prediction! |


Večina teh podatkov so vnaprej določeni pred sestavo načrta, bom še dodatno preveril vsakega. Za zdaj predpostavimo, da so vsi znani in je posek_iglavcev, posek_listavcev res zadnji določen v načrtu.

**!!!** - Nekateri podatki so napačno označeni, določeni atributi naj ne bi nikoli smeli bit določeni glede na oznako drugih atributov (npr. smer = NEGA DEBELJAKA ->  sklep ne sme biti RAHEL (v bazi je takih sestojev 40)). Za take primere se bom pozanimal, da jih pravilno označimo in izločimo.

___

### Sosednost sestojev:

Iz geometrijskih podatkov smo sestavli border-based sosednost sestojev. Unikatno so identificirani po šifri(node_id) -> CONCAT(ggo,odsek,sestoj) ali po ID(id), ki je označen povrsti od 0 do N-1.

Sosednost DB(5.1GB): [adjacency_db](https://drive.google.com/file/d/1CYLelPeP2p0VUPWXUh00pLzenra5a7_b/view?usp=drive_link)

Vsebuje  4 tabele:
- **joined layer** <- raw edge podatki (za debugging)
- **nodes** <- 347.338 sestojev z node_id in id (vključuje ***isolated nodes***)
- **isolated nodes** <- 4.871 izoliranih sestojev, ki so brez edges
- **edges** <- 1.877.718 directional povezav med sestoji s šifro in indexi (node1,node2, n1_id,n2_id)
- ostale tabele v DB so brez predmetne

![Prikaz k-sosednosti](docs/prikaz_sosednosti_v0.png)

___

Potrebno bo pretvorit podatke v smiselno relacijsko bazo:  
(To ni končna verzija, definitivno je treba jo dodelat mankajo tarife...)

![Primer sheme](docs/shema_v2.png)

## RDL-MODEL
Treba sestavit arhitekturo...

### TODO:
- [X] vsaka enota ma tut GEOMETRY property, treba nardit pretvorbo, da iz tega smiselno  dobimo sosednost
- [X] kakšna bo ta sosednost, kajti sestoji so različnih oblik. Se  bo upoštevalo distance, center, border? Treba pomislit in raziskat 
- [ ] določit sosednji k-sestoji embedding (najverjetnje bo max k=2-3 in bo treba embedding nrdit iz njihovih tabel)
- [ ] preveri kolko on average vrednost k (k-sosednost) zajema skupno površino sestojev [ha]
- [ ] mogoč dodamo sosednost na nivoju odseka in celo revirja (za gge je pa njbrz ze overkill)
- [ ] med sloji, bi enote na višjem nivoju lahko delovale kot super node nižjim enotam?
- [ ] iz raw DBja je treba primerno pripravit podatke, to bo odvisno od modela, ki ga bomo uporabli in relacijske sheme
- [ ] dobro bi blo tut vključit podatke območij z lubadrajem, vetrolomom, požarom... Te podatki so very scattered za različna časovna obdobja, treba si pogledat če se da to uporabit iz geometrijskih podatkov
- [ ] treba raziskat kateri model bi bil najbolj učinkovit za naš problem, sj njbrž bo njbulš, da več različnih modelov nrdimo z modifikacijami