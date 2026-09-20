# IF TABLE NOT EXISTS THEN WRITE
#CREATE DATABASE IF NOT EXISTS linkagex;

#IF TABLE EXISTS
CREATE DATABASE linkagex;
use linkagex;

select database();

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    mobile_number VARCHAR(15) NOT NULL,
    age_group VARCHAR(20) NOT NULL,
    profession VARCHAR(50) NOT NULL,
    city_region VARCHAR(100) NOT NULL,
    preferred_language VARCHAR(30),
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('citizen', 'investigator') NOT NULL DEFAULT 'citizen',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


select * from users;



