Example queries from the video:

LLMs Are Databases - So Query Them
https://youtu.be/8Ppw8254nLI?si=80kLyWlgvn-CRp35

STATS;

DESCRIBE "France";

SELECT * FROM EDGE WHERE entity = "France" AND relation = "nationality" LIMIT 5;

DESCRIBE "Einstein";

SELECT * FROM EDGES NEAREST TO "France" AT LAYER 26 LIMIT 10;

SELECT * FROM FEATURES WHERE LAYER = 26;

SELECT * FROM FEATURES WHERE LAYER = 25 AND feature = 5067;

SHOW RELATIONS;

SELECT * FROM EDGES WHERE relation = "capital" LIMIT 100;

SELECT * FROM EDGES NEAREST TO "Einstein" AT LAYER 26 LIMIT 10;

SELECT * FROM ENTITIES LIMIT 20;

SELECT * FROM FEATURES WHERE LAYER = 26 LIMIT 20;

INFER "The captial of France is" TOP 5;

INFER "The captial of Atlantis is" TOP 5;

INSERT INTO EDGES (entity,relation,target) VALUES ("Atlantis","capital","Poseidon");

INFER "The captial of Atlantis is" TOP 5;

DESCRIBE "Atlantis";

COMPILE CURRENT INTO VINDEX "/tmp/atlantis.vindex";


