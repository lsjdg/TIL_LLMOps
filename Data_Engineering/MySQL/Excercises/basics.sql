USE sqldb;

SELECT *
FROM usertbl
WHERE name = '김경호';

SELECT `userID`, name
FROM usertbl
WHERE `birthYear` > 1970
    AND height >= 182;

SELECT name, height
FROM usertbl
WHERE height BETWEEN 180
    AND 183;

SELECT name, addr
FROM usertbl
WHERE addr IN ("경남", "경북");

SELECT name
FROM usertbl
WHERE name LIKE "김%";

SELECT name, height
FROM usertbl
WHERE height > (SELECT height FROM usertbl WHERE name = "김경호");

SELECT name, height
FROM usertbl
WHERE height >= ANY (SELECT height FROM usertbl WHERE addr = "경남");

SELECT DISTINCT addr
FROM usertbl
ORDER BY addr
LIMIT 3;

SELECT `userID`, SUM(amount) "TOTAL AMOUNT"
FROM buytbl
GROUP BY `userID`;

SELECT `userID`, AVG(amount) "AVG AMOUNT"
FROM buytbl
GROUP BY `userID`;

SELECT name, height 
FROM usertbl
WHERE height = (SELECT MAX(height) FROM usertbl)
    OR height = (SELECT MIN(height) FROM usertbl);

SELECT COUNT(mobile1)
FROM usertbl;

