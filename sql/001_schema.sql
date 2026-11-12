-- DiamondIQ schema, MySQL 8.4. Idempotent: safe to run on an existing database.
-- Run by the container on first start (mounted at /docker-entrypoint-initdb.d) and again by
-- `python -m mlb_outlook.db.loader`, which applies it before loading.
--
-- Every prediction row carries its model version and data version, both as foreign keys
-- through model_runs and as plain columns, so a row read on its own still says what made it.

CREATE TABLE IF NOT EXISTS data_versions (
    data_version_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
    sha256          CHAR(64)     NOT NULL,
    source_path     VARCHAR(255) NOT NULL,
    n_rows          INT UNSIGNED NOT NULL,
    n_players       INT UNSIGNED NOT NULL,
    first_season    SMALLINT     NOT NULL,
    last_season     SMALLINT     NOT NULL,
    loaded_at       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (data_version_id),
    UNIQUE KEY uq_data_versions_sha (sha256)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS players (
    player_id INT UNSIGNED NOT NULL,           -- MLBAM id
    name      VARCHAR(128) NOT NULL,
    PRIMARY KEY (player_id),
    KEY ix_players_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- One row per player-season per data version, so a re-pull sits beside the old snapshot
-- instead of overwriting it. Metrics are nullable: missing is never zero.
CREATE TABLE IF NOT EXISTS player_seasons (
    data_version_id INT UNSIGNED NOT NULL,
    player_id       INT UNSIGNED NOT NULL,
    season          SMALLINT     NOT NULL,
    age             TINYINT UNSIGNED NULL,
    pa              SMALLINT UNSIGNED NOT NULL,
    woba            DECIMAL(5,4) NULL,
    xwoba           DECIMAL(5,4) NULL,
    k_pct           DECIMAL(5,4) NULL,
    bb_pct          DECIMAL(5,4) NULL,
    avg_ev          DECIMAL(5,2) NULL,
    barrel_pct      DECIMAL(5,4) NULL,
    hard_hit_pct    DECIMAL(5,4) NULL,
    PRIMARY KEY (data_version_id, player_id, season),
    KEY ix_player_seasons_season (data_version_id, season),
    CONSTRAINT fk_ps_data FOREIGN KEY (data_version_id) REFERENCES data_versions (data_version_id),
    CONSTRAINT fk_ps_player FOREIGN KEY (player_id) REFERENCES players (player_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- One fitted configuration on one fold (or the production forecast).
CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id    INT UNSIGNED NOT NULL AUTO_INCREMENT,
    model_version   VARCHAR(64)  NOT NULL,     -- e.g. ridge_rel+resid_q_pa@<commit>
    data_version_id INT UNSIGNED NOT NULL,
    run_kind        ENUM('backtest', 'holdout', 'production') NOT NULL,
    fold            VARCHAR(16)  NOT NULL,
    config          VARCHAR(16)  NOT NULL,     -- base / extended
    train_targets   VARCHAR(128) NOT NULL,     -- comma-separated seasons
    target_season   SMALLINT     NOT NULL,
    features_through SMALLINT    NOT NULL,
    git_commit      CHAR(40)     NULL,
    generated_at    DATETIME     NULL,
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_run_id),
    UNIQUE KEY uq_model_runs (model_version, data_version_id, fold, config),
    CONSTRAINT fk_mr_data FOREIGN KEY (data_version_id) REFERENCES data_versions (data_version_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS predictions (
    model_run_id    INT UNSIGNED NOT NULL,
    player_id       INT UNSIGNED NOT NULL,
    target_season   SMALLINT     NOT NULL,
    model_version   VARCHAR(64)  NOT NULL,
    data_version_id INT UNSIGNED NOT NULL,
    age_target      TINYINT UNSIGNED NULL,
    pa_lag1         SMALLINT UNSIGNED NULL,
    point_woba      DECIMAL(6,5) NOT NULL,
    lo80            DECIMAL(6,5) NULL,
    hi80            DECIMAL(6,5) NULL,
    marcel_woba     DECIMAL(6,5) NULL,
    weighted3_woba  DECIMAL(6,5) NULL,
    threshold_woba  DECIMAL(4,3) NULL,
    p_breakout      DECIMAL(6,5) NULL,
    p_stable        DECIMAL(6,5) NULL,
    p_regression    DECIMAL(6,5) NULL,
    -- Outcome columns, filled only once the target season is over (backtests).
    actual_woba     DECIMAL(5,4) NULL,
    actual_pa       SMALLINT UNSIGNED NULL,
    PRIMARY KEY (model_run_id, player_id, target_season),
    KEY ix_predictions_player (player_id, target_season),
    CONSTRAINT fk_pr_run FOREIGN KEY (model_run_id) REFERENCES model_runs (model_run_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_pr_player FOREIGN KEY (player_id) REFERENCES players (player_id),
    CONSTRAINT fk_pr_data FOREIGN KEY (data_version_id) REFERENCES data_versions (data_version_id),
    CONSTRAINT ck_pr_interval CHECK (lo80 IS NULL OR hi80 IS NULL OR lo80 <= hi80),
    CONSTRAINT ck_pr_probs CHECK (
        p_breakout IS NULL OR p_stable IS NULL OR p_regression IS NULL
        OR ABS(p_breakout + p_stable + p_regression - 1) < 0.0001
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Long format: one metric value per (run, table, model, slice, metric).
CREATE TABLE IF NOT EXISTS evaluation_results (
    model_run_id INT UNSIGNED NOT NULL,
    source       VARCHAR(32)  NOT NULL,        -- point / interval_coverage / probabilities / ...
    model        VARCHAR(32)  NOT NULL,        -- model or label the metric is about
    slice        VARCHAR(64)  NOT NULL,        -- '' for the whole fold
    metric       VARCHAR(32)  NOT NULL,
    value        DOUBLE       NULL,
    PRIMARY KEY (model_run_id, source, model, slice, metric),
    CONSTRAINT fk_er_run FOREIGN KEY (model_run_id) REFERENCES model_runs (model_run_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
