# Sustainable Forestry Planning with Graph Neural Networks

Naš glavni cilj bo zgraditi GNN model, ki bo na podlagi meta podatkov odsekov za sestoje ter strukture k-sosednih sestojev lahko sam napovedal določene atribute sestojev. Model bo tako lahko prispeval k učinkovitejši in bolj konsistentni pripravi prihodnjih gozdno gospodarskih načrtov.


## Dataset:

Website: [Pregledovalnik podatkov o gozdovih](https://prostor.zgs.gov.si/pregledovalnik/) 

Full raw dataset encompassing the entire country:  

- [50k_all_odseki.sqlite](https://drive.google.com/file/d/1-k_Y9iBYT9Aj8qa6Ns19J-wyWem4eqIh/view?usp=drive_link) (290mb)
- [350k_all_sestoji.sqlite](https://drive.google.com/file/d/1B5DAeCr2gHqvvHo4pdXN9eu8m2iEm7Vg/view?usp=drive_link) (673mb)  



Parsed and pre-processed dataset, ready for GNN training:  

[forest_db.sqlite](https://drive.google.com/file/d/1f3VSCSu_NWmmrRq_ExB0XNmTrIZOsHQU/view?usp=drive_link) (2.45gb)

It includes 5 tables:  
- **nodes** <-- 347.338 nodes with their id name and index 
- **edges** <-- an edge list containing 1.877.718 directed edges
- **isolated_nodes** <-- a list of 4.871 nodes that have no edges
- **sestoji_attr** <-- attribute table for each node
- **odseki_attr** <-- 53.403 regional node attributes


Forest unit hierarchy:  
GGO > Krajevne enote > GGE > Revirji > Odseki > Sestoji

State of the raw dataset before pre-processing:
![DB Data](docs/current_db_shema.png)

### Odsek (Regional attributes):  
    [ ] - represents the number of categories, or the unit of measurement used for the attribute  

| Attribute | Description |
|-----------|-------------|
| ggo | gozdno gospodarkso območje |
| ke | krajevna enota |
| gge | gozdno gospodarksa enota |
| revir | … |
| odsek | … |
| povrsina[ha] | area in hectars |
| gojitveni razred ggo [125] | needs to be double checked and IDs to be fixed |
| gojitveni razred gge [455] | this one needs serious clean up as well as ID fixes |
| kategorija gozda [4] | večnamenski, GPN brez ukrepov, varovalni... |
| ohranjenost gozda [4] | ohranjeni, spremenjeni... |
| polozaj pokrajine [4] | ravnina, pobočje, greben, vznožje |
| relief [9] | grebenasto, vrtačastvo, kotanjastvo... |
| lega [9] | cardinal direction of the slope (S, J, V, Z, SZ, JV...) |
| nagib[˚] | steepness |
| nadm. višina[m] (min,max) | min and max altitude |
| kamnina [39] | apnenec, dolomit, fliš, diluvialna ilovica... |
| delež kamnitosti[%] | delež kamnine, ki je označen zgoraj (preveri) |
| delež skalovitosti[%] | delež skalovja, ki pokriva tla |
| tarife drevesnih vrst[%] | kakovost dreves oz. nek rank |
| odprtost[%] | odprtost zaradi vlak (vlaka = gozdna "cesta") |
| odprtost za gurs | to baje vključuje odprtost zarad vlak in cest nasplošno (preveri) |
| pozarna ogrozenost [5] | majhna ogroženost, srednja ogroženost... |
| intenzivnost gospodarjenja [6] | small-intensity, medium-intensity... |
| rastiščni tip [129] | needs serious clean up as well as ID fixes |
| rastiščni koeficient | check what this value means |
| (vezani ogljik, letni ponor ogljika) | mogoče relevantno mogoče ne? |


### Sestoj (Node attributes):

| Attribute | Description |
|-----------|-------------|
| odsek | regional attribute identifier |
| sestoj | smallest forest unit, the node itself |
| povrsina [ha] | area of the forest unit |
| razvojna faza [11] | mladovje, dvojni sloj, debeljak... |
| gojitvena smer [21] | nega debeljaka, končni posek, ekocelica... |
| zasnova [5] | bogata, pomankljiva... |
| sklep [6] | normalen, rahel, tesen... |
| negovanost [5] | nenegovan, pomankljivo negovan, negovan sestoj... |
| površina pomladka [ha] | … |
| pomladek zasnova [5] | slaba, dobra, bogata... |
| sestava gozda | jelke[%], bukve[%], mehki listavci[%] … |
| sestava mladja | jelke_m[%], bukve_m[%], mehki listavci_m[%] … |
| zaloga iglavcev [m^3] | trenutna zaloga iglavcev v sestoju |
| zaloga listavcev [m^3] | trenutna zaloga listavcev v sestoju |
| posek iglavcev | naš prediction! 0-15000 |
| posek listavcev | naš prediction! 0-15000 |


Večina teh podatkov so vnaprej določeni pred sestavo načrta, bom še dodatno preveril vsakega. Za zdaj predpostavimo, da so vsi znani in je posek_iglavcev, posek_listavcev res zadnji določen v načrtu.

**!!!** - Nekateri podatki so napačno označeni, določeni atributi naj ne bi nikoli smeli bit določeni glede na oznako drugih atributov (npr. smer = NEGA DEBELJAKA ->  sklep ne sme biti RAHEL (v bazi je takih sestojev 40)). Za take primere se bom pozanimal, da jih pravilno označimo in izločimo.

___

### Sosednost sestojev:

From geometric data we were able to construct a border based neighborhood for every node. They are uniquely identified by their string id (node_id) or their index (id) in the parsed dataset.

![Prikaz k-sosednosti](docs/prikaz_sosednosti_v0.png)

___

Potrebno bo pretvorit podatke v smiselno relacijsko bazo:  
(To ni končna verzija, definitivno je treba jo dodelat mankajo tarife...)

![Primer sheme](docs/shema_v2.png)
  

## Model Architecture:
We will most likely develop GraphSAGE and GAT.  

___  

## TODO:
- [X] ~~vsaka enota ma tut GEOMETRY property, treba nardit pretvorbo, da iz tega smiselno  dobimo sosednost~~
- [X] ~~kakšna bo ta sosednost, kajti sestoji so različnih oblik. Se  bo upoštevalo distance, center, border? Treba pomislit in raziskat~~ 
- [ ] lahko bi dodali edge weight based na length of the border med sestoji (GAT)
- [X] ~~določit sosednji k-sestoji embedding (najverjetnje bo max k=2-3 in bo treba embedding nrdit iz njihovih tabel)~~
- [ ] preveri kolko on average vrednost k (k-sosednost) zajema skupno površino sestojev [ha]
- [ ] mogoč dodamo sosednost na nivoju odseka in celo revirja (za gge je pa njbrz ze overkill)
- [ ] med sloji, bi enote na višjem nivoju lahko delovale kot super node nižjim enotam? al samo dodamo regional atribute kot dodaten node embedding? reku je prof nj oboje sprobamo
- [ ] potrebno je izločit faulty podatke iz baze (negativne vrednosti, nemogoči sestoji...), za to sem že dodal db_fix_v0.sql file
- [ ] lahko tut eksperimentiramo da ene atribute spustimo in jih ne uporabimo, mogoče bo bolši?
- [X] ~~iz raw DBja je treba primerno pripravit podatke, to bo odvisno od relacijske sheme ter modela, ki ga bomo uporabli~~
- [ ] splitamo DB na train,valid,test set (njbrz bomo po 396 revirjih splital: 75-12.5-12.5 --> 296-50-50)
- [ ] simpl python code za different seed splitanje DB-ja (po revirih, pa morda se po odsekih/revirih)
- [ ] dobro bi blo tut vključit podatke območij z lubadarjem, vetrolomom, požarom... Te podatki so very scattered za različna časovna obdobja, treba si pogledat če se da to uporabit iz geometrijskih podatkov
- [X] ~~treba raziskat kateri model bi bil najbolj učinkovit za naš problem, sj njbrž bo njbulš, da več različnih modelov nrdimo z modifikacijami~~
- [X] ~~basic model GraphSAGE~~ 
- [X] ~~dodelaj GraphSAGE, da vključi sestavo gozda~~
- [ ] vključi še attribute odsekov
- [ ] basic model GAT (al pa kerga druzga)
