-- =============================================================================
-- AI-Powered Student Performance Analytics System — Database Schema
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Table 1: Synthetic cohort students (15 students, 2024-2025 marks)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY, -- e.g. S001
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS marks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL REFERENCES students (student_id),
    year INTEGER NOT NULL, -- 2024 or 2025
    subject TEXT NOT NULL, -- Math | Physics | CS | English | Statistics
    mark REAL -- NULL if missing/imputed
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL REFERENCES students (student_id),
    subject TEXT NOT NULL,
    predicted_mark REAL,
    model_used TEXT, -- LinearRegression | DecisionTree | RandomForest
    created_at TEXT DEFAULT(datetime('now'))
);

-- ---------------------------------------------------------------------------
-- Table 2: VLE dataset — 160 real student instances (pass/fail classification)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vle_students (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,

-- Behaviour features
gender INTEGER, -- 0=male, 1=female
age REAL,
logins INTEGER, -- Count of module area logins
total_hours REAL, -- Total hours in module area
pct_avg_hours REAL, -- % of average total hours in module area
presence_count INTEGER, -- Count of presence in compulsory sessions
absence_count INTEGER, -- Count of absence in compulsory sessions
pct_attended REAL, -- % of compulsory learning sessions attended

-- Neighbourhood / contextual features
attending_from_home INTEGER, -- 1=home, 0=alternative address
distance_to_uni_km REAL,
polar4_quintile INTEGER,
polar3_quintile INTEGER,
adult_he_2001_quintile INTEGER,
adult_he_2011_quintile INTEGER,
tundra_msoa_quintile INTEGER,
tundra_lsoa_quintile INTEGER,
gaps_gcse_quintile INTEGER,
gaps_gcse_ethnicity_quintile INTEGER,
uni_connect_target_ward INTEGER,

-- Label
label INTEGER NOT NULL, -- 1=fail, 0=pass

-- Source split tracking
split                       TEXT DEFAULT 'original'  -- original | train | test
);

CREATE TABLE IF NOT EXISTS vle_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vle_student_id INTEGER NOT NULL REFERENCES vle_students (id),
    predicted_label INTEGER, -- 0=pass, 1=fail
    predicted_proba REAL, -- probability of fail
    model_used TEXT,
    created_at TEXT DEFAULT(datetime('now'))
);