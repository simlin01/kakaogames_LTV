#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3-Stage LTV Prediction Pipeline

This script trains a three-stage model to predict user LTV.
- Stage 1 (Binary: Payer vs Non-Payer): CatBoost + LightGBM + XGBoost (hard voting).
- Stage 2 (Binary: Whale vs Non-Whale): CatBoost + LightGBM + TabPFN (hard voting).
- Stage 3 (Regression on Payers): Two-headed CatBoost + LightGBM + TabPFN Regressors (non-whale/whale).

The script is designed to be executed from the command line, with arguments to
specify the execution stage and random seed.
"""

import os
import sys
import math
import random
import warnings
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import joblib
import logging
from datetime import datetime
import uuid
import traceback
import json

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.integration import XGBoostPruningCallback, LightGBMPruningCallback

import torch

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    _HAS_TABPFN = True
except Exception as e:
    _HAS_TABPFN = False
    warnings.warn(f"TabPFN import failed: {e}. Stage2/3 will skip TabPFN.")

from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix, mean_absolute_error, mean_squared_error,roc_auc_score
)

# ------Test mode------
TEST_FRACTION = 0.10  # test_mode에서 사용할 샘플링 비율

# =====================================================================================
# ---- 1. CONFIGURATION & PATHS
# =====================================================================================

# main.py가 있는 폴더 (ver1_stage1)
SCRIPT_DIR = Path(__file__).resolve().parent

# 데이터 폴더 경로 (ver1_stage1에서 한 단계 위로 올라가 Data 폴더로 이동)
DATA_DIR = SCRIPT_DIR.parent / "Data"

# 결과물 저장 경로
# ver1_stage1 폴더 내에 생성됩니다.
RESULTS_DIR = SCRIPT_DIR / "seed_results_stage1"
RESULTS_DIR.mkdir(exist_ok=True) 

# 이 부분은 DATA_DIR이 올바르게 정의되면 자동으로 잘 작동합니다.
DATA_PATHS = {
    "train": str(DATA_DIR / "train_df_5days.parquet"),
    "val":   str(DATA_DIR / "val_df_5days.parquet"),
    "test":  str(DATA_DIR / "test_df_5days.parquet"),
    "train_robust": str(DATA_DIR / "train_df_5days_robust.parquet"),
    "val_robust":   str(DATA_DIR / "val_df_5days_robust.parquet"),
    "test_robust":  str(DATA_DIR / "test_df_5days_robust.parquet"),
}

# --- Global Settings ---
DEFAULT_SEED = 2025

#################### 본인 시드로 수정수정!!!!!!!!!!!1##################
SEEDS = list(range(2021, 2024))  # 10 seeds: 2021..2030
#################### 본인 시드로 수정수정!!!!!!!!!!!1##################

TARGET_COL = "PAY_AMT_SUM"
ID_COL = "PLAYERID"
WHALE_Q = 0.95  # top 5% among TRAIN payers

# --- Model & Tuning Settings ---
CUT_STEP = 0.01
DELTA_AROUND = 0.15
USE_OPTUNA = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CAT_TASK_PARAMS = {} # GPU 사용 시 {"task_type": "GPU"}
CAT_TASK_PARAMS = {"task_type": "GPU"} if DEVICE == "cuda" else {}

# --- Stage 1 Fixed Hyperparameters ---
RUN_LR = 0.05
LGBM_FIXED = dict(
    objective="binary", learning_rate=RUN_LR, subsample=0.1, min_child_samples=20,
    reg_alpha=0.1, reg_lambda=0.1, verbosity=-1,
)
LGBM_MAX_DEPTH_RANGE = (7, 14)
LGBM_N_EST_RANGE     = (800, 1600)

XGB_FIXED = dict(
    objective="binary:logistic", eval_metric="auc", learning_rate=RUN_LR, subsample=0.1,
    reg_alpha=0.1, reg_lambda=0.1, tree_method="hist", max_bin=256,
)
XGB_MAX_DEPTH_RANGE = (7, 14)
XGB_N_EST_RANGE     = (800, 1600)

CAT_FIXED = dict(
    loss_function="Logloss", eval_metric="F1", learning_rate=RUN_LR, verbose=0,
)
CAT_DEPTH_RANGE = (7, 14)
CAT_ITER_RANGE  = (800, 1600)

OPTUNA_TRIALS = {
    "stage1": 50,
    "stage2": 30,
    "stage3_nw": 30,
    "stage3_w": 30,
}

OPTUNA_SEED = DEFAULT_SEED

# =====================================================================================
# ---- 2. UTILITY FUNCTIONS (Original notebook code)
# =====================================================================================

def _select_params(model_name: str, params: dict) -> dict:
    """
    CSV 길이를 줄이기 위해 모델별 핵심 파라미터만 추려 반환.
    """
    if model_name == "lgbm":
        keep = [
            "n_estimators","learning_rate","num_leaves","max_depth",
            "min_child_samples","subsample","colsample_bytree","reg_alpha","reg_lambda",
            "random_state"
        ]
    elif model_name == "xgb":
        keep = [
            "n_estimators","learning_rate","max_depth","min_child_weight",
            "subsample","colsample_bytree","reg_alpha","reg_lambda","scale_pos_weight",
            "random_state","n_jobs"
        ]
    else:  # cat
        keep = [
            "depth","iterations","learning_rate","l2_leaf_reg",
            "bagging_temperature","random_seed","class_weights"
        ]
    return {k: params.get(k) for k in keep if k in params}


# --- TabPFN Helper ---
import inspect

def _construct_tabpfn(cls, device: str, seed: int, n_ens: int):
    try:
        sig = inspect.signature(cls.__init__)
        kw = {}
        # 공통적으로 시도 가능한 인자만 선별
        if "device" in sig.parameters:
            kw["device"] = device
        # 앙상블 크기 이름 호환
        if "N_ensemble_configurations" in sig.parameters:
            kw["N_ensemble_configurations"] = n_ens
        elif "n_estimators" in sig.parameters:
            kw["n_estimators"] = n_ens
        # 시드 이름 호환
        if "seed" in sig.parameters:
            kw["seed"] = seed
        elif "random_state" in sig.parameters:
            kw["random_state"] = seed
        # 위 인자들이 하나도 안 맞아도 최소한 기본 생성은 시도
        return cls(**kw) if kw else cls()
    except Exception:
        # 최후의 보루: 아무 인자 없이 생성
        return cls()

def _make_tabpfn_classifier(device: str, seed: int, n_ens: int = 16):
    if not _HAS_TABPFN:
        return None
    return _construct_tabpfn(TabPFNClassifier, device=device, seed=seed, n_ens=n_ens)

def _make_tabpfn_regressor(device: str, seed: int, n_ens: int = 16):
    if not _HAS_TABPFN:
        return None
    return _construct_tabpfn(TabPFNRegressor, device=device, seed=seed, n_ens=n_ens)

# --- Metrics & Preprocessing ---
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)
    diff[denom == 0] = 0.0
    return float(np.mean(diff) * 100)

class OrdinalCategoryEncoder:
    def __init__(self):
        self.maps: Dict[str, Dict] = {}
        self.cols: List[str] = []

    def fit(self, df: pd.DataFrame, cat_cols: List[str]):
        self.cols = list(cat_cols)
        for c in self.cols:
            # 카테고리 목록을 안전하게 확보
            cats = pd.Series(df[c].astype("category").cat.categories)
            self.maps[c] = {cat: i for i, cat in enumerate(cats)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.cols:
            if c in out.columns:
                mapping = self.maps[c]
                # ⚠️ Categorical → object로 풀고 dict.get으로 안전 매핑
                s = out[c].astype(object)
                out[c] = s.apply(lambda v: mapping.get(v, -1)).astype(np.int32)
        return out

class XGBCompat:
    """
    xgboost 버전 무관하게 early stopping + predict_proba 제공.
    내부적으로 sklearn API가 아니라 xgb.train(DMatrix) 사용.
    """
    def __init__(self, **params):
        self.params = params.copy()
        self.booster_ = None
        self.best_ntree_limit_ = None
        self._num_boost_round = int(self.params.pop("n_estimators", 100))

    def _to_train_params(self):
        p = self.params.copy()
        # 이름 매핑
        if "random_state" in p and "seed" not in p:
            p["seed"] = p.pop("random_state")
        if "n_jobs" in p and "nthread" not in p:
            p["nthread"] = p.pop("n_jobs")
        # 안전 기본값
        p.setdefault("objective", "binary:logistic")
        p.setdefault("eval_metric", p.get("eval_metric", "auc"))
        return p

    def fit(self, X_tr, y_tr, X_va, y_va, early_stopping_rounds=200, verbose_eval=False):
        import xgboost as xgb
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va, label=y_va)
        train_params = self._to_train_params()
        self.booster_ = xgb.train(
            train_params,
            dtr,
            num_boost_round=self._num_boost_round,
            evals=[(dtr, "train"), (dva, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )
        # best_ntree_limit은 버전에 따라 없을 수 있음
        self.best_ntree_limit_ = getattr(self.booster_, "best_ntree_limit", None)
        return self

    def predict_proba(self, X):
        import xgboost as xgb
        d = xgb.DMatrix(X)
        if self.best_ntree_limit_:
            p1 = self.booster_.predict(d, ntree_limit=self.best_ntree_limit_)
        else:
            p1 = self.booster_.predict(d)
        p1 = np.asarray(p1, dtype=float).reshape(-1)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

# --- Data Loading ---
action_trash_list = ['길드_하우스 대여', '캐시 상점_아이템 삭제', '길드_가입 신청', '계정_로그인', '클래스_잠금',
                     '길드_설정 변경', '성장_레벨 다운', '성장_스킬 습득', '그로아_소환 확정 대기 변경', '아이템 컬렉션_추가',
                     '그로아_소환', '탈것_스킬 설정', '퀘스트_보상 미리보기 삭제', '캐시 상점_아이템 추가', '길드_생성', '제작_제작',
                     '클래스_소환 확정 대기 생성', '계정_로그아웃', '길드_적대 등록 취소', '길드_등급', '길드_동맹 신청 취소', '보스전_필드 보스',
                     '길드_동맹 신청', '탈것_추가', '탈것_소환 확정 대기 변경', '퀘스트_포기', '그로아_소환 확정 대기 생성', '성장_레벨 업',
                     '캐시 상점_월드 추가', '사망 불이익_경험치', '캐시 상점_캐시 상점에서 재화로 구매', '퀘스트_보상 미리보기', '캐릭터_생성',
                     '클래스_소환 확정 대기 변경', '길드_적대 등록', '던젼_충전', '스탯_설정', '기믹_등짐', '클래스_소환 확정 대기 삭제', '그로아_소환 확정 대기 삭제',
                     '성장_상태 변화 습득', '성장_죽음', '제작_추가', '퀘스트_의뢰 갱신', '길드_지원자 제거', '캐시 상점_캐릭터 추가', '길드_동맹 파기', '워프_갱신',
                     '워프_삭제', '클래스_추가', '길드_가입', '길드_동맹 신청 확인', '보스전_월드 보스', '퀘스트_완료', '길드_해체', '탈것_잠금', '캐시 상점_계정 추가',
                     '워프_생성', '워프_순간이동 사용', '성장_경험치 손실', '퀘스트_의뢰', '퀘스트_수락', '탈것_등록', '퀘스트_수행', '길드_경험치 획득', '그로아_잠금',
                     '캐시 상점_구매 나이 변경', '길드_동맹 신청 거절', '탈것_소환 확정 대기 생성', '클래스_변경', '탈것_소환 확정 대기 삭제', '길드_탈퇴', '사망 불이익_아이템',
                     '길드_출석', '그로아_추가']

action_list = ['PLAYERID','계정', '그로아', '기믹', '길드', '던젼', '보스전',
               '사망 불이익', '성장', '스탯', '아이템 컬렉션', '워프', '제작',
               '캐릭터', '캐시 상점', '퀘스트', '클래스', '탈것']

def load_pre_split(is_test_mode: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # base
    df_train = pd.read_parquet(DATA_PATHS["train"], engine="pyarrow")
    df_val   = pd.read_parquet(DATA_PATHS["val"],   engine="pyarrow")
    df_test  = pd.read_parquet(DATA_PATHS["test"],  engine="pyarrow")
    # robust (슬림 컬럼만 유지)
    rb_train = pd.read_parquet(DATA_PATHS["train_robust"], engine="pyarrow")
    rb_val   = pd.read_parquet(DATA_PATHS["val_robust"],   engine="pyarrow")
    rb_test  = pd.read_parquet(DATA_PATHS["test_robust"],  engine="pyarrow")

    base_train = df_train.drop(columns=action_trash_list, errors="ignore")
    base_val   = df_val.drop(columns=action_trash_list, errors="ignore")
    base_test  = df_test.drop(columns=action_trash_list, errors="ignore")

    rb_train = rb_train[action_list]
    rb_val   = rb_val[action_list]
    rb_test  = rb_test[action_list]

    train = base_train.merge(rb_train, on="PLAYERID", how="left")
    val   = base_val.merge(rb_val,     on="PLAYERID", how="left")
    test  = base_test.merge(rb_test,   on="PLAYERID", how="left")

    # NAT_CD 카테고리 정합성(있으면)
    if "NAT_CD" in train.columns:
        train["NAT_CD"] = train["NAT_CD"].astype("category")
        cats = train["NAT_CD"].cat.categories
        if "NAT_CD" in val.columns:
            val["NAT_CD"] = pd.Categorical(val["NAT_CD"], categories=cats)
        if "NAT_CD" in test.columns:
            test["NAT_CD"] = pd.Categorical(test["NAT_CD"], categories=cats)

    if is_test_mode:
        logging.info("⚡ Test mode enabled. Sampling fixed counts (train=2000, val=1000, test=1000).")
        train = train.sample(n=min(2000, len(train)), random_state=DEFAULT_SEED)
        val   = val.sample(n=min(1000, len(val)),   random_state=DEFAULT_SEED)
        test  = test.sample(n=min(1000, len(test)), random_state=DEFAULT_SEED)

    def _ratio(df):
        return float((df[TARGET_COL] > 0).mean()) if len(df) > 0 else 0.0
    logging.info(f"🔎 Loaded pre-split | payer ratio — train:{_ratio(train):.4f}  val:{_ratio(val):.4f}  test:{_ratio(test):.4f}")
    return train, val, test

def _sanitize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str).str.replace(r"\s+", "_", regex=True)
    return out

def build_features(df: pd.DataFrame, target_col: str, drop_cols: List[str]):
    cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = [c for c in cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    return df[cols].copy(), cols, cat_cols

def fit_imputer(train_df: pd.DataFrame):
    num_cols = [c for c in train_df.columns if train_df[c].dtype != 'object' and not str(train_df[c].dtype).startswith('category')]
    med = train_df[num_cols].median(numeric_only=True)
    return num_cols, med

def apply_imputer(df: pd.DataFrame, num_cols: list[str], med: pd.Series):
    df = df.copy()
    df[num_cols] = df[num_cols].fillna(med)
    # OrdinalCategoryEncoder로 범주는 이미 정수화됨(미등록은 -1), 혹시 남은 NaN이 있으면 0으로:
    df = df.fillna(0)
    return df

# =====================================================================================
# ---- 3. STAGE-SPECIFIC TRAINING LOGIC (All functions from notebook)
# =====================================================================================
# ------------------------
# Cutoff search
# ------------------------

def _search_cutoff_grid(y_true, proba, center: float, delta: float = DELTA_AROUND, step: float = CUT_STEP, metric: str = "f1"):
    lo = max(0.0, center - delta)
    hi = min(1.0, center + delta)
    grid = np.arange(lo, hi + 1e-9, step)
    y_true = np.asarray(y_true).astype(int)
    scores = {}
    best_t, best_s = 0.5, -1.0
    for t in grid:
        y_hat = (proba >= t).astype(int)
        if metric == "f1":
            s = f1_score(y_true, y_hat) if (y_hat.sum()>0 and y_true.sum()>0) else 0.0
        elif metric == "balanced_acc":
            tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
            tpr = tp / (tp + fn + 1e-12)
            tnr = tn / (tn + fp + 1e-12)
            s = 0.5*(tpr+tnr)
        else:
            s = f1_score(y_true, y_hat)
        scores[float(t)] = float(s)
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, scores


def tune_cutoff(y_true, proba, strategy: str, train_pos_prior: float, metric: str = "f1"):
    if strategy == "prior":
        center = float(np.clip(train_pos_prior, 0.05, 0.95))
    elif strategy == "reweight":
        center = 0.5
    else:
        center = 0.5
    return _search_cutoff_grid(y_true, proba, center=center, metric=metric)

# ------------------------
# Optuna - classification
# ------------------------

def _tune_lgbm_cls(X_tr, y_tr, X_va, y_va, stage_key: str, strategy: str, train_pos_prior: float, size="large"):
    # size: "large"(stage1) | "small"(stage2)
    n_trials = OPTUNA_TRIALS[stage_key]
    if size == "large":
        bounds = dict(
            n_estimators=(800, 4000),
            learning_rate=(0.01, 0.1),
            num_leaves=(63, 255),
            max_depth=(-1, 14),
            min_child_samples=(20, 200),
            subsample=(0.6, 1.0),
            colsample_bytree=(0.6, 1.0),
            reg_alpha=(0.0, 5.0),
            reg_lambda=(0.0, 10.0),
        )
    else:  # small (stage2)
        bounds = dict(
            n_estimators=(500, 2500),
            learning_rate=(0.01, 0.15),
            num_leaves=(31, 127),
            max_depth=(-1, 10),
            min_child_samples=(10, 200),
            subsample=(0.6, 1.0),
            colsample_bytree=(0.6, 1.0),
            reg_alpha=(0.0, 8.0),
            reg_lambda=(0.0, 15.0),
        )

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", *bounds["n_estimators"]),
            learning_rate=trial.suggest_float("learning_rate", *bounds["learning_rate"], log=True),
            num_leaves=trial.suggest_int("num_leaves", *bounds["num_leaves"]),
            max_depth=trial.suggest_int("max_depth", *bounds["max_depth"]),
            min_child_samples=trial.suggest_int("min_child_samples", *bounds["min_child_samples"]),
            subsample=trial.suggest_float("subsample", *bounds["subsample"]),
            colsample_bytree=trial.suggest_float("colsample_bytree", *bounds["colsample_bytree"]),
            reg_alpha=trial.suggest_float("reg_alpha", *bounds["reg_alpha"]),
            reg_lambda=trial.suggest_float("reg_lambda", *bounds["reg_lambda"]),
            objective="binary",
            random_state=SEED,
            n_jobs=max(1, (os.cpu_count() or 8)//4),
            verbosity=-1,
            force_row_wise=True,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(200, verbose=False)])
        proba = model.predict_proba(X_va)[:,1]
        t_opt, _ = tune_cutoff(y_va, proba, strategy=strategy, train_pos_prior=train_pos_prior)
        pred = (proba >= t_opt).astype(int)
        return f1_score(y_va, pred)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    best.update(dict(objective="binary", random_state=SEED, n_jobs=max(1, (os.cpu_count() or 8)//4)))
    # 🔇 불필요 로그/오버헤드 축소
    best.update({"verbosity": -1,         # info 로그 끄기
                 "force_row_wise": True,  # "Auto-choosing row-wise..." 안내문 제거 + 고정
    })
    return best


def _tune_xgb_cls(X_tr, y_tr, X_va, y_va, stage_key: str, strategy: str, train_pos_prior: float,
                  scale_pos_weight=1.0, size="large"):
    n_trials = OPTUNA_TRIALS[stage_key]
    if size == "large":
        bounds = dict(
            n_estimators=(800, 4000),
            learning_rate=(0.01, 0.2),
            max_depth=(4, 10),
            min_child_weight=(1.0, 10.0),
            subsample=(0.6, 1.0),
            colsample_bytree=(0.6, 1.0),
            reg_alpha=(0.0, 5.0),
            reg_lambda=(0.0, 10.0),
        )
    else:
        bounds = dict(
            n_estimators=(500, 2500),
            learning_rate=(0.01, 0.2),
            max_depth=(3, 8),
            min_child_weight=(1.0, 12.0),
            subsample=(0.6, 1.0),
            colsample_bytree=(0.6, 1.0),
            reg_alpha=(0.0, 8.0),
            reg_lambda=(0.0, 15.0),
        )

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", *bounds["n_estimators"]),
            learning_rate=trial.suggest_float("learning_rate", *bounds["learning_rate"], log=True),
            max_depth=trial.suggest_int("max_depth", *bounds["max_depth"]),
            min_child_weight=trial.suggest_float("min_child_weight", *bounds["min_child_weight"]),
            subsample=trial.suggest_float("subsample", *bounds["subsample"]),
            colsample_bytree=trial.suggest_float("colsample_bytree", *bounds["colsample_bytree"]),
            reg_alpha=trial.suggest_float("reg_alpha", *bounds["reg_alpha"]),
            reg_lambda=trial.suggest_float("reg_lambda", *bounds["reg_lambda"]),
            objective="binary:logistic",
            random_state=SEED,
            n_jobs=max(1, (os.cpu_count() or 8)//4),
            tree_method="hist",
            max_bin=256,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
        )
        model = XGBCompat(**params)
        model.fit(X_tr, y_tr, X_va, y_va, early_stopping_rounds=200, verbose_eval=False)
        proba = model.predict_proba(X_va)[:, 1]
        t_opt, _ = tune_cutoff(y_va, proba, strategy=strategy, train_pos_prior=train_pos_prior)
        pred = (proba >= t_opt).astype(int)
        return f1_score(y_va, pred)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    best.update(dict(objective="binary:logistic", random_state=SEED, n_jobs=max(1, (os.cpu_count() or 8)//4),
                     tree_method="hist", max_bin=256, scale_pos_weight=scale_pos_weight, eval_metric="auc",))
    return best


def _tune_cat_cls(X_tr, y_tr, X_va, y_va, cat_cols_idx, stage_key: str, strategy: str,
                  train_pos_prior: float, class_weights=None, size="large"):
    n_trials = OPTUNA_TRIALS[stage_key]
    bounds = dict(iterations=(800, 4000), depth=(5, 10)) if size=="large" else dict(iterations=(600, 2500), depth=(4, 8))

    def objective(trial):
        params = dict(
            depth=trial.suggest_int("depth", *bounds["depth"]),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 12.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 3.0),
            random_strength=trial.suggest_float("random_strength", 0.0, 2.0),
            iterations=trial.suggest_int("iterations", *bounds["iterations"]),
            loss_function="Logloss",
            eval_metric="F1",
            random_seed=SEED,
            verbose=0,
        )
        if class_weights is not None:
            params["class_weights"] = class_weights

        model = CatBoostClassifier(**params, **CAT_TASK_PARAMS, od_type="Iter", od_wait=200)
        pool_tr = Pool(X_tr, y_tr, cat_features=cat_cols_idx or None)
        pool_va = Pool(X_va, y_va, cat_features=cat_cols_idx or None)

        model.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)
        proba = model.predict_proba(pool_va)[:, 1]
        t_opt, _ = tune_cutoff(y_va, proba, strategy=strategy, train_pos_prior=train_pos_prior)
        return f1_score(y_va, (proba >= t_opt).astype(int))

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    best.update(dict(loss_function="Logloss", eval_metric="F1", random_seed=SEED, verbose=0))
    if class_weights is not None:
        best["class_weights"] = class_weights
    return best

def _tune_lgbm_md_ne_fixedgrid(X_tr, y_tr, X_va, y_va, stage_key: str, strategy: str, train_pos_prior: float,
                               lgb_train=None, lgb_valid=None):
    assert lgb_train is not None and lgb_valid is not None, "Pass prebuilt lgb.Dataset"
    n_trials = OPTUNA_TRIALS[stage_key]

    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=DEFAULT_SEED),
                                pruner=MedianPruner(n_warmup_steps=5, n_min_trials=3))

    def objective(trial):
        # ✅ 고정값 + 속도 옵션
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'subsample': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
            'verbosity': -1,            # 🔇
            'random_state': DEFAULT_SEED,
            'n_jobs': -1,
            'force_row_wise': True,     # ⚡
        }

        # ✅ 튜닝 값
        params['max_depth'] = trial.suggest_int("max_depth", *LGBM_MAX_DEPTH_RANGE)
        params['n_estimators'] = trial.suggest_int("n_estimators", *LGBM_N_EST_RANGE)
        params['colsample_bytree'] = trial.suggest_categorical("colsample_bytree", [0.1, 0.3, 0.5])

        # 🔪 Optuna pruner + ⏱️ 조기 종료
        pruning_callback = LightGBMPruningCallback(trial, "auc", valid_name="valid")
        early_stop_cb = lgb.early_stopping(200, verbose=False)
        log_cb = lgb.log_evaluation(0)

        # ✅ n_estimators를 num_boost_round로 반영
        num_boost_round = int(params.pop('n_estimators'))

        booster = lgb.train(
            params,
            lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_valid],
            valid_names=["valid"],
            callbacks=[pruning_callback, early_stop_cb, log_cb],
        )

        proba = booster.predict(X_va, num_iteration=booster.best_iteration)
        t_opt, _ = tune_cutoff(y_va, proba, strategy=strategy, train_pos_prior=train_pos_prior)
        return f1_score(y_va, (proba >= t_opt).astype(int))

    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_params.update({
        'objective': 'binary',
        'learning_rate': 0.05,
        'subsample': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_samples': 20,
        'verbosity': -1,
        'random_state': DEFAULT_SEED,
        'n_jobs': -1,
        'force_row_wise': True,
    })
    return best_params


def _tune_xgb_md_ne_fixedgrid(X_tr, y_tr, X_va, y_va, stage_key: str, strategy: str, train_pos_prior: float,
                              scale_pos_weight=1.0, dtr=None, dva=None):
    assert dtr is not None and dva is not None, "Pass prebuilt DMatrix"
    n_trials = OPTUNA_TRIALS[stage_key]

    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=DEFAULT_SEED),
                                pruner=MedianPruner(n_warmup_steps=5, n_min_trials=3))

    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.05,
            'subsample': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'tree_method': 'hist',    # GPU 쓰면 'gpu_hist'로 교체 가능
            'max_bin': 256,
            'random_state': DEFAULT_SEED,
            'n_jobs': -1,
            'scale_pos_weight': scale_pos_weight,
        }
        params['max_depth'] = trial.suggest_int("max_depth", *XGB_MAX_DEPTH_RANGE)
        params['n_estimators'] = trial.suggest_int("n_estimators", *XGB_N_EST_RANGE)
        params['colsample_bytree'] = trial.suggest_categorical("colsample_bytree", [0.1, 0.3, 0.5])

        pruning_callback = XGBoostPruningCallback(trial, "valid-auc")

        num_boost_round = params.pop('n_estimators')

        booster = xgb.train(
            params,
            dtr,
            num_boost_round=num_boost_round,
            evals=[(dtr, "train"), (dva, "valid")],
            callbacks=[pruning_callback],
            verbose_eval=False,
            early_stopping_rounds=200,     # ✅ 추가: 트라이얼 내 조기 종료
        )

        try:
            proba = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
        except TypeError:
            # iteration_range 미지원 버전 대비
            best_ntree = getattr(booster, "best_ntree_limit", None)
            if best_ntree is not None:
                proba = booster.predict(dva, ntree_limit=best_ntree)
            else:
                proba = booster.predict(dva)
        t_opt, _ = tune_cutoff(y_va, proba, strategy=strategy, train_pos_prior=train_pos_prior)
        return f1_score(y_va, (proba >= t_opt).astype(int))

    study.optimize(objective, n_trials=n_trials)

    # study에서 찾은 최적값으로 최종 파라미터 구성
    best_params = study.best_params
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.05,
        'subsample': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'tree_method': 'hist',
        'max_bin': 256,
        'random_state': DEFAULT_SEED,
        'n_jobs': -1,
        'scale_pos_weight': scale_pos_weight,
    })
    return best_params


def _tune_cat_md_ne(X_tr, y_tr, X_va, y_va, cat_cols_idx, stage_key: str, strategy: str,
                    train_pos_prior: float, class_weights=None):
    n_trials = OPTUNA_TRIALS[stage_key]
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=DEFAULT_SEED),
                                pruner=MedianPruner(n_warmup_steps=5, n_min_trials=3))

    def objective(trial):
        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'F1',
            'learning_rate': 0.05,
            'verbose': 0,
            'random_seed': DEFAULT_SEED,
            'depth': trial.suggest_int('depth', *CAT_DEPTH_RANGE),
        }
        if class_weights is not None:
            params["class_weights"] = class_weights

        model = CatBoostClassifier(**params, **CAT_TASK_PARAMS)
        pool_tr = Pool(X_tr, y_tr, cat_features=cat_cols_idx or None)
        pool_va = Pool(X_va, y_va, cat_features=cat_cols_idx or None)

        model.fit(pool_tr, eval_set=pool_va, use_best_model=True,
                  early_stopping_rounds=200, verbose=False)

        trial.set_user_attr("best_iteration", model.get_best_iteration())

        proba = model.predict_proba(pool_va)[:, 1]
        t_opt, _ = tune_cutoff(y_va, proba, strategy=strategy, train_pos_prior=train_pos_prior)
        return f1_score(y_va, (proba >= t_opt).astype(int))

    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_params['iterations'] = study.best_trial.user_attrs.get('best_iteration', CAT_ITER_RANGE[1])
    best_params.update({
        'loss_function': 'Logloss',
        'eval_metric': 'F1',
        'learning_rate': 0.05,
        'verbose': 0,
        'random_seed': DEFAULT_SEED,
    })
    if class_weights is not None:
        best_params["class_weights"] = class_weights
    return best_params

# ------------------------
# Optuna - Regression
# ------------------------

def _tune_lgbm_reg(X_tr, y_tr, X_va, y_va, stage_key: str, size="mid"):
    # size: "mid"(~10k; non-whale) | "tiny"(~100; whale)
    n_trials = OPTUNA_TRIALS[stage_key]
    if size == "mid":
        bounds = dict(
            n_estimators=(800, 6000),
            learning_rate=(0.01, 0.1),
            num_leaves=(63, 255),
            max_depth=(-1, 12),
            min_child_samples=(10, 200),
            subsample=(0.6, 1.0),
            colsample_bytree=(0.6, 1.0),
            reg_alpha=(0.0, 5.0),
            reg_lambda=(0.0, 10.0),
        )
    else:  # tiny
        bounds = dict(
            n_estimators=(300, 1500),
            learning_rate=(0.01, 0.2),
            num_leaves=(15, 63),
            max_depth=(3, 8),
            min_child_samples=(5, 50),
            subsample=(0.7, 1.0),
            colsample_bytree=(0.7, 1.0),
            reg_alpha=(0.0, 10.0),
            reg_lambda=(1.0, 20.0),
        )

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", *bounds["n_estimators"]),
            learning_rate=trial.suggest_float("learning_rate", *bounds["learning_rate"], log=True),
            num_leaves=trial.suggest_int("num_leaves", *bounds["num_leaves"]),
            max_depth=trial.suggest_int("max_depth", *bounds["max_depth"]),
            min_child_samples=trial.suggest_int("min_child_samples", *bounds["min_child_samples"]),
            subsample=trial.suggest_float("subsample", *bounds["subsample"]),
            colsample_bytree=trial.suggest_float("colsample_bytree", *bounds["colsample_bytree"]),
            reg_alpha=trial.suggest_float("reg_alpha", *bounds["reg_alpha"]),
            reg_lambda=trial.suggest_float("reg_lambda", *bounds["reg_lambda"]),
            objective="mae",
            random_state=SEED,
            n_jobs=max(1, (os.cpu_count() or 8)//4),
            verbosity=-1,
            force_row_wise=True,
        )
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="l1",
                  callbacks=[lgb.early_stopping(200, verbose=False)])
        pred = model.predict(X_va)
        return mean_absolute_error(y_va, pred)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    best.update(dict(objective="mae", random_state=SEED, n_jobs=max(1, (os.cpu_count() or 8)//4)))
    # 🔇 동일 적용
    best.update({"verbosity": -1,
                 "force_row_wise": True,
    })
    return best


def _tune_cat_reg(X_tr, y_tr, X_va, y_va, cat_cols_idx, stage_key: str, size="mid"):
    n_trials = OPTUNA_TRIALS[stage_key]
    if size == "mid":
        bounds = dict(iterations=(1500, 6000), depth=(6, 10), l2=(1.0, 10.0))
    else:  # tiny
        bounds = dict(iterations=(600, 2000), depth=(4, 7), l2=(3.0, 15.0))

    def objective(trial):
        params = dict(
            depth=trial.suggest_int("depth", *bounds["depth"]),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", *bounds["l2"]),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 3.0),
            random_strength=trial.suggest_float("random_strength", 0.0, 2.0),
            iterations=trial.suggest_int("iterations", *bounds["iterations"]),
            loss_function="MAE",
            random_seed=SEED,
            verbose=0,
        )
        model = CatBoostRegressor(**params)
        pool_tr = Pool(X_tr, y_tr, cat_features=cat_cols_idx or None)
        pool_va = Pool(X_va, y_va, cat_features=cat_cols_idx or None)
        model.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)
        pred = model.predict(pool_va)
        return mean_absolute_error(y_va, pred)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    best.update(dict(loss_function="MAE", random_seed=SEED, verbose=0))
    return best

# ------------------------
# Stage1 – Cat + LGBM + XGB (hard vote) with dual-strategy selection
# ------------------------

def train_stage1_models(
    X_tr, y_tr, X_va, y_va, cat_cols_idx, _unused_strategy, pos_prior,
    lgb_train, lgb_valid, dtr, dva,
    pool_tr=None, pool_va=None
):
    # --- CatBoost (Pool) : 필요할 때만 생성하고 재사용
    if pool_tr is None:
        pool_tr = Pool(X_tr, y_tr, cat_features=cat_cols_idx or None)
    if pool_va is None:
        pool_va = Pool(X_va, y_va, cat_features=cat_cols_idx or None)

    # CatBoost 튜닝 + 학습
    cat_params = _tune_cat_md_ne(
        X_tr, y_tr, X_va, y_va, cat_cols_idx,
        stage_key="stage1", strategy="prior",
        train_pos_prior=pos_prior, class_weights=None
    )
    cat1 = CatBoostClassifier(**cat_params, **CAT_TASK_PARAMS, od_type="Iter", od_wait=200)
    cat1.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)
    p_cat = cat1.predict_proba(pool_va)[:, 1]

    # LightGBM
    lgb_params = _tune_lgbm_md_ne_fixedgrid(
        X_tr, y_tr, X_va, y_va, stage_key="stage1",
        strategy="prior", train_pos_prior=pos_prior,
        lgb_train=lgb_train, lgb_valid=lgb_valid
    )
    lgbm1 = lgb.LGBMClassifier(**lgb_params)
    lgbm1.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    )
    p_lgb = lgbm1.predict_proba(X_va)[:, 1]

    # XGBoost
    xgb_params = _tune_xgb_md_ne_fixedgrid(
        X_tr, y_tr, X_va, y_va, stage_key="stage1",
        strategy="prior", train_pos_prior=pos_prior,
        scale_pos_weight=1.0, dtr=dtr, dva=dva
    )
    xgb1 = XGBCompat(**xgb_params)
    xgb1.fit(X_tr, y_tr, X_va, y_va, early_stopping_rounds=100, verbose_eval=False)
    p_xgb = xgb1.predict_proba(X_va)[:, 1]

    # 컷오프 선택 + 하드보팅
    preds = {"cat": p_cat, "lgbm": p_lgb, "xgb": p_xgb}
    cut_prior, cut_rew = {}, {}
    for k, p in preds.items():
        cut_prior[k] = tune_cutoff(y_va, p, strategy="prior",    train_pos_prior=pos_prior)[0]
        cut_rew[k]   = tune_cutoff(y_va, p, strategy="reweight", train_pos_prior=pos_prior)[0]

    yhat_prior = hard_vote(preds, cut_prior)
    yhat_rew   = hard_vote(preds, cut_rew)
    f1_prior = f1_score(y_va, yhat_prior)
    f1_rew   = f1_score(y_va, yhat_rew)

    if f1_prior >= f1_rew:
        return {"cat": cat1, "lgbm": lgbm1, "xgb": xgb1}, preds, cut_prior, "prior", f1_prior
    else:
        return {"cat": cat1, "lgbm": lgbm1, "xgb": xgb1}, preds, cut_rew,   "reweight", f1_rew


def hard_vote(preds: Dict[str, np.ndarray], cutoffs: Dict[str, float]) -> np.ndarray:
    votes = []
    for k, p in preds.items():
        t = cutoffs[k]
        votes.append((p >= t).astype(int))
    votes = np.column_stack(votes)
    return (votes.sum(axis=1) >= int(math.ceil(votes.shape[1]/2))).astype(int)

# ------------------------
# Stage2 – Cat + LGBM + TabPFN (hard vote) with dual-strategy selection
# ------------------------

def train_stage2_models(X_tr, y_tr, X_va, y_va, cat_cols_idx, _unused_strategy, pos_prior):
    def _weights(y):
        n_pos = int(y.sum()); n_neg = int(len(y) - n_pos)
        w_pos = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
        return 1.0, w_pos

    best = {"f1": -1.0, "strategy": None}
    for strat in ["prior", "reweight"]:
        models = {}; preds = {}; cutoffs = {}
        w_neg, w_pos = _weights(y_tr)

        # CatBoost
        if USE_OPTUNA:
            cat_params = _tune_cat_cls(X_tr, y_tr, X_va, y_va, cat_cols_idx,
                                       stage_key="stage2", strategy=strat, train_pos_prior=pos_prior,
                                       class_weights=[w_neg, w_pos] if strat=="reweight" else None, size="small")
        else:
            cat_params = dict(depth=6, learning_rate=0.05, iterations=2500, loss_function="Logloss",
                              eval_metric="F1", random_seed=SEED, verbose=0)
            if strat == "reweight":
                cat_params.update(class_weights=[w_neg, w_pos])
        # CatBoost (Pool)
        cat2 = CatBoostClassifier(**cat_params, **CAT_TASK_PARAMS)
        pool_tr = Pool(X_tr, y_tr, cat_features=cat_cols_idx or None)
        pool_va = Pool(X_va, y_va, cat_features=cat_cols_idx or None)
        cat2.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)
        p_cat = cat2.predict_proba(pool_va)[:, 1]
        t_cat, _ = tune_cutoff(y_va, p_cat, strategy=strat, train_pos_prior=pos_prior)
        models["cat"], preds["cat"], cutoffs["cat"] = cat2, p_cat, t_cat

        # LightGBM
        if USE_OPTUNA:
            lgb_params = _tune_lgbm_cls(X_tr, y_tr, X_va, y_va, stage_key="stage2",
                                        strategy=strat, train_pos_prior=pos_prior, size="small")
            if strat == "reweight":
                lgb_params.update(class_weight={0:w_neg, 1:w_pos})
        else:
            lgb_params = dict(n_estimators=4500, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                              objective="binary", random_state=SEED, n_jobs=max(1, (os.cpu_count() or 8)//4))
            if strat == "reweight":
                lgb_params.update(class_weight={0:w_neg, 1:w_pos})
        lgbm2 = lgb.LGBMClassifier(**lgb_params)
        lgbm2.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        p_lgb = lgbm2.predict_proba(X_va)[:,1]
        t_lgb, _ = tune_cutoff(y_va, p_lgb, strategy=strat, train_pos_prior=pos_prior)
        models["lgbm"], preds["lgbm"], cutoffs["lgbm"] = lgbm2, p_lgb, t_lgb

        # TabPFN은 튜닝 없음(고정)
        if _HAS_TABPFN:
            logging.info(f"[TabPFN] device set to: {DEVICE}")   # ← 여기!
            tab2 = _make_tabpfn_classifier(device=DEVICE, seed=SEED, n_ens=16)
            tab2.fit(np.asarray(X_tr), np.asarray(y_tr))
            p_tab = tab2.predict_proba(np.asarray(X_va))[:, 1]
            t_tab, _ = tune_cutoff(y_va, p_tab, strategy=strat, train_pos_prior=pos_prior)
            models["tab"], preds["tab"], cutoffs["tab"] = tab2, p_tab, t_tab

        yhat = hard_vote(preds, cutoffs)
        f1_val = f1_score(y_va, yhat)
        if f1_val > best["f1"]:
            best.update(f1=f1_val, strategy=strat, models=models, preds=preds, cutoffs=cutoffs)

    return best["models"], best["preds"], best["cutoffs"], best["strategy"], best["f1"]

# ------------------------
# Stage3 – TWO-HEAD regressors (whale / non-whale) with mean & median ensembles
# ------------------------

def train_stage3_regressors_twohead(X_pay_tr, y_amt_tr, y_whale_tr,
                                    X_pay_va, y_amt_va, y_whale_va, cat_cols_idx):
    """
    Stage3 TWO-HEAD 회귀 (non-whale / whale)
    - 각 헤드: CatBoostRegressor + LGBMRegressor + (옵션) TabPFNRegressor
    - Optuna로 Cat/LGBM 튜닝(이미 상단 헬퍼 사용), TabPFN은 고정 파라미터
    - VAL에서 두 헤드 각각 예측 후 payer별로 붙여 mean/median 앙상블 반환
    """
    tr0 = np.where(y_whale_tr == 0)[0]; tr1 = np.where(y_whale_tr == 1)[0]
    va0 = np.where(y_whale_va == 0)[0]; va1 = np.where(y_whale_va == 1)[0]

    # ---------- non-whale 헤드 (~1e4) ----------
    if USE_OPTUNA:
        cat_params0 = _tune_cat_reg(X_pay_tr.iloc[tr0], y_amt_tr[tr0],
                                    X_pay_va.iloc[va0], y_amt_va[va0],
                                    cat_cols_idx, stage_key="stage3_nw", size="mid")
        lgb_params0 = _tune_lgbm_reg(X_pay_tr.iloc[tr0], y_amt_tr[tr0],
                                     X_pay_va.iloc[va0], y_amt_va[va0],
                                     stage_key="stage3_nw", size="mid")
    else:
        cat_params0 = dict(depth=8, learning_rate=0.05, iterations=5000,
                           loss_function="MAE", random_seed=SEED, verbose=0)
        lgb_params0 = dict(n_estimators=6000, learning_rate=0.03, num_leaves=127,
                           subsample=0.8, colsample_bytree=0.8, objective="mae",
                           random_state=SEED, n_jobs=max(1, (os.cpu_count() or 8)//4))

    catR0 = CatBoostRegressor(**cat_params0, **CAT_TASK_PARAMS)
    lgbR0 = lgb.LGBMRegressor(**lgb_params0)

    catR0.fit(
        Pool(X_pay_tr.iloc[tr0], y_amt_tr[tr0], cat_features=cat_cols_idx or None),
        eval_set=Pool(X_pay_va.iloc[va0], y_amt_va[va0], cat_features=cat_cols_idx or None),
        use_best_model=True, verbose=False
    )
    lgbR0.fit(X_pay_tr.iloc[tr0], y_amt_tr[tr0],
              eval_set=[(X_pay_va.iloc[va0], y_amt_va[va0])], eval_metric="l1",
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

    # + TabPFN (옵션)
    tabR0 = None
    if _HAS_TABPFN:
        logging.info(f"[TabPFN] device set to: {DEVICE}")   # ← 여기!
        tabR0 = _make_tabpfn_regressor(device=DEVICE, seed=SEED, n_ens=16)
        tabR0.fit(np.asarray(X_pay_tr.iloc[tr0]), np.asarray(y_amt_tr[tr0]))

    # ---------- whale 헤드 (~1e2) ----------
    if USE_OPTUNA:
        cat_params1 = _tune_cat_reg(X_pay_tr.iloc[tr1], y_amt_tr[tr1],
                                    X_pay_va.iloc[va1], y_amt_va[va1],
                                    cat_cols_idx, stage_key="stage3_w", size="tiny")
        lgb_params1 = _tune_lgbm_reg(X_pay_tr.iloc[tr1], y_amt_tr[tr1],
                                     X_pay_va.iloc[va1], y_amt_va[va1],
                                     stage_key="stage3_w", size="tiny")
    else:
        cat_params1 = dict(depth=6, learning_rate=0.05, iterations=1500,
                           loss_function="MAE", random_seed=SEED, verbose=0, l2_leaf_reg=8.0)
        lgb_params1 = dict(n_estimators=1200, learning_rate=0.05, num_leaves=31,
                           subsample=0.9, colsample_bytree=0.9, objective="mae",
                           random_state=SEED, n_jobs=max(1, (os.cpu_count() or 8)//4), reg_lambda=10.0)

    catR1 = CatBoostRegressor(**cat_params1, **CAT_TASK_PARAMS)
    lgbR1 = lgb.LGBMRegressor(**lgb_params1)

    catR1.fit(
        Pool(X_pay_tr.iloc[tr1], y_amt_tr[tr1], cat_features=cat_cols_idx or None),
        eval_set=Pool(X_pay_va.iloc[va1], y_amt_va[va1], cat_features=cat_cols_idx or None),
        use_best_model=True, verbose=False
    )
    lgbR1.fit(X_pay_tr.iloc[tr1], y_amt_tr[tr1],
              eval_set=[(X_pay_va.iloc[va1], y_amt_va[va1])], eval_metric="l1",
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

    # + TabPFN (옵션)
    tabR1 = None
    if _HAS_TABPFN:
        logging.info(f"[TabPFN] device set to: {DEVICE}")   # ← 여기!
        tabR1 = _make_tabpfn_regressor(device=DEVICE, seed=SEED, n_ens=16)
        tabR1.fit(np.asarray(X_pay_tr.iloc[tr1]), np.asarray(y_amt_tr[tr1]))

    # ---------- VAL 예측(두 헤드) ----------
    def _pred(models_head, X_local):
        parts = [
            models_head["cat"].predict(Pool(X_local, cat_features=cat_cols_idx or None)),
            models_head["lgbm"].predict(X_local),
        ]
        if "tab" in models_head:
            parts.append(models_head["tab"].predict(np.asarray(X_local)))
        P = np.column_stack(parts)
        return np.mean(P, axis=1), np.median(P, axis=1)

    models = {
        "nonwhale": {"cat": catR0, "lgbm": lgbR0},
        "whale":    {"cat": catR1, "lgbm": lgbR1},
    }
    if _HAS_TABPFN:
        models["nonwhale"]["tab"] = tabR0
        models["whale"]["tab"]    = tabR1

    n_va = len(X_pay_va)
    va_mean = np.zeros(n_va, dtype=float)
    va_med  = np.zeros(n_va, dtype=float)

    if len(va0):
        m0, md0 = _pred(models["nonwhale"], X_pay_va.iloc[va0])
        va_mean[va0] = m0; va_med[va0] = md0
    if len(va1):
        m1, md1 = _pred(models["whale"], X_pay_va.iloc[va1])
        va_mean[va1] = m1; va_med[va1] = md1

    return models, {"mean": va_mean, "median": va_med}

# ------------------------
# End-to-end
# ------------------------

def run_pipeline(seed: int = 2025):
    global SEED, OPTUNA_SEED
    SEED = seed
    OPTUNA_SEED = seed
    np.random.seed(SEED)
    random.seed(SEED)

    # Load
    train, val, test = load_pre_split()

    # Stage1 labels
    y_tr = (train[TARGET_COL] > 0).astype(int)
    y_va = (val[TARGET_COL] > 0).astype(int)

    # Features
    drop_cols = [ID_COL, TARGET_COL]
    Xtr_raw, feat_cols, cat_cols = build_features(train, TARGET_COL, drop_cols)
    Xva_raw = val[feat_cols].copy(); Xte_raw = test[feat_cols].copy()

    enc = OrdinalCategoryEncoder().fit(Xtr_raw, cat_cols)
    Xtr = enc.transform(Xtr_raw); Xva = enc.transform(Xva_raw); Xte = enc.transform(Xte_raw)
    cat_cols_idx = [Xtr.columns.get_loc(c) for c in cat_cols if c in Xtr.columns]

    num_cols, med = fit_imputer(Xtr)
    Xtr = apply_imputer(Xtr, num_cols, med)
    Xva = apply_imputer(Xva, num_cols, med)
    Xte = apply_imputer(Xte, num_cols, med)

    # ↓ 컬럼명 공백 제거 (경고 억제용)
    Xtr = _sanitize_cols(Xtr)
    Xva = _sanitize_cols(Xva)
    Xte = _sanitize_cols(Xte)

    lgb_train = lgb.Dataset(Xtr, label=y_tr, free_raw_data=False)
    lgb_valid = lgb.Dataset(Xva, label=y_va, reference=lgb_train, free_raw_data=False)

    dtr = xgb.DMatrix(Xtr, label=y_tr)
    dva = xgb.DMatrix(Xva, label=y_va)

    # Stage1 – dual strategy auto-select
    pos_prior1 = float(y_tr.mean())
    m1, p1, t1, strat1, f1_best1 = train_stage1_models(
        Xtr, y_tr, Xva, y_va, cat_cols_idx, None, pos_prior1,
        lgb_train=lgb_train, lgb_valid=lgb_valid, dtr=dtr, dva=dva
    )
    yhat1_va = hard_vote(p1, t1)
    f1_1 = f1_score(y_va, yhat1_va); pr_1 = precision_score(y_va, yhat1_va); rc_1 = recall_score(y_va, yhat1_va)
    logging.info(f"[Seed {SEED}] [Stage1|VAL] F1={f1_1:.4f} Prec={pr_1:.4f} Rec={rc_1:.4f} PosPrior={pos_prior1:.4f} | ChosenStrategy={strat1} (VoteF1={f1_best1:.4f})")

    # Stage2 (payers only) – dual strategy auto-select
    tr_pay_idx = np.where(y_tr == 1)[0]; va_pay_idx = np.where(y_va == 1)[0]
    y_tr_pay_amt = train.loc[train[TARGET_COL] > 0, TARGET_COL].values
    whale_cut = float(np.quantile(y_tr_pay_amt, WHALE_Q))

    y2_tr = (train.loc[train[TARGET_COL] > 0, TARGET_COL].values >= whale_cut).astype(int)
    y2_va = (val.loc[val[TARGET_COL] > 0, TARGET_COL].values >= whale_cut).astype(int)

    X2_tr = Xtr.iloc[tr_pay_idx].reset_index(drop=True)
    X2_va = Xva.iloc[va_pay_idx].reset_index(drop=True)

    pos_prior2 = float(y2_tr.mean())
    m2, p2, t2, strat2, f1_best2 = train_stage2_models(X2_tr, y2_tr, X2_va, y2_va, cat_cols_idx, None, pos_prior2)
    yhat2_va = hard_vote(p2, t2)
    f1_2 = f1_score(y2_va, yhat2_va); pr_2 = precision_score(y2_va, yhat2_va); rc_2 = recall_score(y2_va, yhat2_va)
    logging.info(f"[Seed {SEED}] [Stage2|VAL] F1={f1_2:.4f} Prec={pr_2:.4f} Rec={rc_2:.4f} PosPrior={pos_prior2:.4f} WhaleCut={whale_cut:.1f} | ChosenStrategy={strat2} (VoteF1={f1_best2:.4f})")

    # Stage3 (two-head regression on payers)
    y3_tr = train.loc[train[TARGET_COL] > 0, TARGET_COL].values.astype(float)
    y3_va = val.loc[val[TARGET_COL] > 0, TARGET_COL].values.astype(float)

    m3, p3_va = train_stage3_regressors_twohead(X2_tr, y3_tr, y2_tr, X2_va, y3_va, y2_va, cat_cols_idx)

    # ---- Calibration constants from VAL for actual ensembles ----
    Ybar = float(np.mean(y3_va))
    Xbar_mean = float(np.mean(p3_va["mean"])) if np.mean(p3_va["mean"]) != 0 else 1e-9
    Xbar_median = float(np.mean(p3_va["median"])) if np.mean(p3_va["median"]) != 0 else 1e-9
    c_mean = Ybar / Xbar_mean
    c_median = Ybar / Xbar_median
    var_mean = float(np.sum((y3_va - p3_va["mean"])**2))
    var_median = float(np.sum((y3_va - p3_va["median"])**2))

    mae_va_mean = mean_absolute_error(y3_va, p3_va["mean"]);  smape_va_mean = smape(y3_va, p3_va["mean"])
    mae_va_med  = mean_absolute_error(y3_va, p3_va["median"]); smape_va_med  = smape(y3_va, p3_va["median"])
    logging.info(f"[Seed {SEED}] [Stage3|VAL|Mean]   MAE={mae_va_mean:.2f} SMAPE={smape_va_mean:.2f}")
    logging.info(f"[Seed {SEED}] [Stage3|VAL|Median] MAE={mae_va_med:.2f}  SMAPE={smape_va_med:.2f}")

    # ---------- TEST INFERENCE ----------
    # Stage1
    pool_te = Pool(Xte, cat_features=cat_cols_idx or None)
    p1_te = {
        "cat":  m1["cat"].predict_proba(pool_te)[:, 1],
        "lgbm": m1["lgbm"].predict_proba(Xte)[:, 1],
        "xgb":  m1["xgb"].predict_proba(Xte)[:, 1],
    }
    yhat1_te = hard_vote(p1_te, t1)

    # Stage2 within predicted payers
    te_pay_idx = np.where(yhat1_te == 1)[0]
    X2_te = Xte.iloc[te_pay_idx].reset_index(drop=True)
    pool_te_pay = Pool(X2_te, cat_features=cat_cols_idx or None)
    p2_te = {
        "cat":  m2["cat"].predict_proba(pool_te_pay)[:, 1],
        "lgbm": m2["lgbm"].predict_proba(X2_te)[:, 1],
    }
    if "tab" in m2:
        p2_te["tab"] = m2["tab"].predict_proba(np.asarray(X2_te))[:, 1]
    yhat2_te = hard_vote(p2_te, t2)  # 1=whale among TEST payers

    # Stage3 routing
    te0_local = np.where(yhat2_te == 0)[0]; te1_local = np.where(yhat2_te == 1)[0]

    def _pred_head(models, X_local):
        parts = [
            models["cat"].predict(Pool(X_local, cat_features=cat_cols_idx or None)),
            models["lgbm"].predict(X_local),
        ]
        if "tab" in models:
            parts.append(models["tab"].predict(np.asarray(X_local)))
        P = np.column_stack(parts)
        return np.mean(P, axis=1), np.median(P, axis=1)

    mean0, med0 = _pred_head(m3["nonwhale"], X2_te.iloc[te0_local]) if len(te0_local)>0 else (np.array([]), np.array([]))
    mean1, med1 = _pred_head(m3["whale"],     X2_te.iloc[te1_local]) if len(te1_local)>0 else (np.array([]), np.array([]))

    final_mean = np.zeros(len(Xte), dtype=float)
    final_median = np.zeros(len(Xte), dtype=float)
    if len(te0_local)>0:
        final_mean[te_pay_idx[te0_local]] = mean0
        final_median[te_pay_idx[te0_local]] = med0
    if len(te1_local)>0:
        final_mean[te_pay_idx[te1_local]] = mean1
        final_median[te_pay_idx[te1_local]] = med1

    whale_mask_te = np.zeros(len(Xte), dtype=int)
    whale_mask_te[te_pay_idx] = yhat2_te

    out_mean = pd.DataFrame({ID_COL: test[ID_COL].values, "pred_pay_amt_sum": final_mean, "pred_is_payer": yhat1_te, "pred_is_whale": whale_mask_te})
    out_median = pd.DataFrame({ID_COL: test[ID_COL].values, "pred_pay_amt_sum": final_median, "pred_is_payer": yhat1_te, "pred_is_whale": whale_mask_te})

    # --- Calibration-based TEST predictions (actual ensemble) ---
    out_mean_calib = out_mean.copy();   out_mean_calib["pred_pay_amt_sum"] *= c_mean
    out_median_calib = out_median.copy(); out_median_calib["pred_pay_amt_sum"] *= c_median

    return {
        "val_metrics": {
            "stage1": {"f1": f1_1, "precision": pr_1, "recall": rc_1, "pos_prior": pos_prior1, "chosen_strategy": strat1},
            "stage2": {"f1": f1_2, "precision": pr_2, "recall": rc_2, "pos_prior": pos_prior2, "whale_cut": whale_cut, "chosen_strategy": strat2},
            "stage3": {
                "mean":   {"mae": mae_va_mean, "smape": smape_va_mean},
                "median": {"mae": mae_va_med,  "smape": smape_va_med },
            },
        },
        "cutoffs": {"stage1": t1, "stage2": t2},
        "chosen_strategies": {"stage1": strat1, "stage2": strat2},
        "whale_cut": whale_cut,
        "pred_test_mean": out_mean,
        "pred_test_median": out_median,
        "pred_test_mean_calibrated": out_mean_calib,
        "pred_test_median_calibrated": out_median_calib,
        "calibration_stage3": {"mean": {"c": c_mean, "variation": var_mean}, "median": {"c": c_median, "variation": var_median}},
    }

def run_all_seeds(seeds=SEEDS):
    all_results = []
    for sd in seeds:
        logging.info(f"================ Seed {sd} ================")
        res = run_pipeline(seed=sd)
        all_results.append(res)
    # Aggregate TEST predictions across seeds (element-wise)
    preds_mean = [r["pred_test_mean"]["pred_pay_amt_sum"].values for r in all_results]
    preds_median = [r["pred_test_median"]["pred_pay_amt_sum"].values for r in all_results]
    payers = [r["pred_test_mean"]["pred_is_payer"].values for r in all_results]
    whales = [r["pred_test_mean"]["pred_is_whale"].values for r in all_results]

    preds_mean = np.column_stack(preds_mean)
    preds_median = np.column_stack(preds_median)
    payers = np.column_stack(payers)
    whales = np.column_stack(whales)

    final_pred_mean = preds_mean.mean(axis=1)
    final_pred_median = np.median(preds_median, axis=1)

    # Majority vote across seeds for masks
    final_is_payer = (payers.sum(axis=1) >= int(math.ceil(payers.shape[1]/2))).astype(int)
    final_is_whale = (whales.sum(axis=1) >= int(math.ceil(whales.shape[1]/2))).astype(int)

    # Build final DataFrames using the first seed's IDs (identical order)
    any_df = all_results[0]["pred_test_mean"]
    df_mean = any_df[["PLAYERID"]].copy()
    df_mean["pred_pay_amt_sum"] = final_pred_mean
    df_mean["pred_is_payer"] = final_is_payer
    df_mean["pred_is_whale"] = final_is_whale

    df_median = any_df[["PLAYERID"]].copy()
    df_median["pred_pay_amt_sum"] = final_pred_median
    df_median["pred_is_payer"] = final_is_payer
    df_median["pred_is_whale"] = final_is_whale

    return {"by_seed": all_results, "final_mean": df_mean, "final_median": df_median, "seeds": list(seeds)}

# ================= Evaluations & Reports =================

def _metrics_basic(y_true, pred):
    mse = mean_squared_error(y_true, pred)
    mae = mean_absolute_error(y_true, pred)
    rmse = float(np.sqrt(mse))
    sm = smape(y_true, pred)
    return mse, mae, rmse, sm

def _fmt2(x):
    if x is None or (isinstance(x, float) and (pd.isna(x) or pd.isnull(x))):
        return "NA"
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)

# === table helpers (SE 포함, 표시 깔끔화) ===
def make_top_groups_table(y, p, P_seeds=None, top_qs=(0.99, 0.97, 0.95, 0.90, 0.80)):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    pos_mask = (y > 0)
    if not pos_mask.any():
        return pd.DataFrame(columns=["상위 퍼센트","샘플 수","MAE","SMAPE","예측평균(SE)","실제 평균"])

    y_pos = y[pos_mask]
    rows = []
    S = None if P_seeds is None else P_seeds.shape[1]
    for q in top_qs:
        thr = np.quantile(y_pos, q)
        idx = np.where((y > 0) & (y >= thr))[0]
        if idx.size == 0:
            continue

        mae = mean_absolute_error(y[idx], p[idx])
        sm  = smape(y[idx], p[idx])
        pred_mean = p[idx].mean()

        if P_seeds is not None:
            m_per_seed = P_seeds[idx].mean(axis=0)   # (S,)
            se = (m_per_seed.std(ddof=1) / np.sqrt(S)) if S and S > 1 else 0.0
            se_str = _fmt2(se)
        else:
            se_str = "NA"

        rows.append({
            "상위 퍼센트": f"{int((1-q)*100)}%",
            "샘플 수": f"{int(idx.size):,}",
            "MAE": _fmt2(mae),
            "SMAPE": _fmt2(sm),
            "예측평균(SE)": f"{_fmt2(pred_mean)} ({se_str})",
            "실제 평균": _fmt2(y[idx].mean()),
        })

    return pd.DataFrame(rows, columns=["상위 퍼센트","샘플 수","MAE","SMAPE","예측평균(SE)","실제 평균"])


def make_quantile_bins_table(y, p, P_seeds=None, bins=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    pos_mask = (y > 0)
    if not pos_mask.any():
        return pd.DataFrame(columns=["Quantile 구간","샘플 수","MAE","SMAPE","예측평균(SE)","실제 평균"])

    y_pos = y[pos_mask]
    qs = np.quantile(y_pos, bins)
    rows = []
    S = None if P_seeds is None else P_seeds.shape[1]
    for i in range(len(qs)-1):
        lo, hi = qs[i], qs[i+1]
        if i < len(qs)-2:
            idx = np.where((y > 0) & (y >= lo) & (y <  hi))[0]
        else:
            idx = np.where((y > 0) & (y >= lo) & (y <= hi))[0]
        if idx.size == 0:
            continue

        mae = mean_absolute_error(y[idx], p[idx])
        sm  = smape(y[idx], p[idx])
        pred_mean = p[idx].mean()

        if P_seeds is not None:
            m_per_seed = P_seeds[idx].mean(axis=0)   # (S,)
            se = (m_per_seed.std(ddof=1) / np.sqrt(S)) if S and S > 1 else 0.0
            se_str = _fmt2(se)
        else:
            se_str = "NA"

        rows.append({
            "Quantile 구간": f"{int(bins[i]*100)}% ~ {int(bins[i+1]*100)}%",
            "샘플 수": f"{int(idx.size):,}",
            "MAE": _fmt2(mae),
            "SMAPE": _fmt2(sm),
            "예측평균(SE)": f"{_fmt2(pred_mean)} ({se_str})",
            "실제 평균": _fmt2(y[idx].mean()),
        })

    return pd.DataFrame(rows, columns=["Quantile 구간","샘플 수","MAE","SMAPE","예측평균(SE)","실제 평균"])

# === 평가 1: 시드별 + 앙상블(보정 전) ======================================
def evaluate_after_seeds(agg_result: Dict):
    from IPython.display import display

    # 테스트셋 로드
    try:
        test_df = pd.read_parquet(DATA_PATHS["test"])
    except Exception:
        logging.info("[WARN] Cannot load test parquet for evaluation.")
        return
    if TARGET_COL not in test_df.columns:
        logging.info("[INFO] Test set has no target column. Skipping evaluation.")
        return

    y = test_df[TARGET_COL].values.reshape(-1)

    # 시드별 예측(Mean path)
    seeds = agg_result.get("seeds", SEEDS)
    P = np.column_stack([
        r["pred_test_mean"]["pred_pay_amt_sum"].values
        for r in agg_result["by_seed"]
    ])  # (N, S)

    # --- Per-seed report ---
    logging.info("="*80)
    logging.info("🔶 Per-Seed Results (각 시드의 단일 모델 예측)")
    for sd, col in zip(seeds, P.T):
        mse, mae, rmse, sm = _metrics_basic(y, col)
        logging.info(f"[Seed {sd}] MAE: {mae:,.2f} | RMSE: {rmse:,.2f} | SMAPE: {sm:.2f}%")
        df_top  = make_top_groups_table(y, col, P_seeds=None)
        df_bins = make_quantile_bins_table(y, col, P_seeds=None)
        logging.info("  · 지출 상위 그룹")
        logging.info(f"\n{df_top.to_string()}")
        logging.info("  · Quantile 구간")
        logging.info(f"\n{df_bins.to_string()}")

    # --- Ensemble mean (보정 전) ---
    ens_mean = P.mean(axis=1)
    mse, mae, rmse, sm = _metrics_basic(y, ens_mean)
    logging.info("="*80)
    logging.info("🔷 FINAL RESULT — Seed Ensemble (MEAN, uncalibrated)")
    logging.info(f"MAE: {mae:,.2f} | RMSE: {rmse:,.2f} | SMAPE: {sm:.2f}%")

    df_top_ens  = make_top_groups_table(y, ens_mean, P_seeds=P)
    df_bins_ens = make_quantile_bins_table(y, ens_mean, P_seeds=P)
    logging.info("\n📈  [지출 상위 그룹 기준] 평균 / SE / MAE / SMAPE")
    logging.info(df_top_ens)
    logging.info("\n📊  [Quantile 구간별] 평균 / SE / MAE / SMAPE")
    logging.info(df_bins_ens)

    return {
        "per_seed": {"metrics": [
            {"seed": int(sd), "MAE": float(_metrics_basic(y, col)[1]),
             "RMSE": float(_metrics_basic(y, col)[2]),
             "SMAPE": float(_metrics_basic(y, col)[3])}
            for sd, col in zip(seeds, P.T)
        ]},
        "ensemble_mean": {
            "metrics": {"MAE": mae, "RMSE": rmse, "SMAPE": sm},
            "df_top": df_top_ens, "df_bins": df_bins_ens
        }
    }

def run_stage1_only(trials=50, seed=2025, is_test_mode=False):
    global SEED, OPTUNA_SEED
    SEED = int(seed)
    OPTUNA_SEED = int(seed)          # Optuna도 시드별로 재현되게
    np.random.seed(SEED)
    random.seed(SEED)

    OPTUNA_TRIALS["stage1"] = int(trials)

    train, val, _ = load_pre_split(is_test_mode=is_test_mode)
    y_tr = (train[TARGET_COL] > 0).astype(int)
    y_va = (val[TARGET_COL] > 0).astype(int)

    drop_cols = [ID_COL, TARGET_COL]
    Xtr_raw, feat_cols, cat_cols = build_features(train, TARGET_COL, drop_cols)
    Xva_raw = val[feat_cols].copy()

    enc = OrdinalCategoryEncoder().fit(Xtr_raw, cat_cols)
    Xtr = enc.transform(Xtr_raw); Xva = enc.transform(Xva_raw)
    cat_cols_idx = [Xtr.columns.get_loc(c) for c in cat_cols if c in Xtr.columns]

    num_cols, med = fit_imputer(Xtr)
    Xtr = apply_imputer(Xtr, num_cols, med); Xva = apply_imputer(Xva, num_cols, med)
    Xtr = _sanitize_cols(Xtr); Xva = _sanitize_cols(Xva)

    # === 재사용 컨테이너(한 번만 생성) ===
    lgb_train = lgb.Dataset(Xtr, label=y_tr, free_raw_data=False)
    lgb_valid = lgb.Dataset(Xva, label=y_va, reference=lgb_train, free_raw_data=False)

    dtr = xgb.DMatrix(Xtr, label=y_tr)
    dva = xgb.DMatrix(Xva, label=y_va)

    pos_prior1 = float(y_tr.mean())
    pool_tr = Pool(Xtr, y_tr, cat_features=cat_cols_idx or None)
    pool_va = Pool(Xva, y_va, cat_features=cat_cols_idx or None)

    models, preds, cutoffs, chosen_strategy, vote_f1 = train_stage1_models(
        Xtr, y_tr, Xva, y_va, cat_cols_idx, None, pos_prior1,
        lgb_train=lgb_train, lgb_valid=lgb_valid, dtr=dtr, dva=dva,
        pool_tr=pool_tr, pool_va=pool_va
    )
    yhat_va = hard_vote(preds, cutoffs)
    f1  = f1_score(y_va, yhat_va)
    prc = precision_score(y_va, yhat_va, zero_division=0)
    rcl = recall_score(y_va, yhat_va, zero_division=0)

    # 🔹 AUC(확률 평균)
    try:
        prob_ens = (preds["cat"] + preds["lgbm"] + preds["xgb"]) / 3.0
        auc = roc_auc_score(y_va, prob_ens)
    except Exception:
        auc = float("nan")

    # 🔹 Best params 추출(핵심 파라미터만)
    best_lgb_all = models["lgbm"].get_params()
    best_xgb_all = models["xgb"].params
    best_cat_all = models["cat"].get_params()

    best_lgb = _select_params("lgbm", best_lgb_all)
    best_xgb = _select_params("xgb",  best_xgb_all)
    best_cat = _select_params("cat",  best_cat_all)

    logging.info(f"[Seed {SEED}] [Stage1|VAL] "
                 f"F1={f1:.4f} | Precision={prc:.4f} | Recall={rcl:.4f} | AUC={auc:.4f} "
                 f"| PosPrior={pos_prior1:.4f} | Strategy={chosen_strategy}")
    logging.info(f"[Best LGBM]: {best_lgb}")
    logging.info(f"[Best XGB ]: {best_xgb}")
    logging.info(f"[Best CAT ]: {best_cat}")

    return {
        "seed": SEED,
        "Xtr":Xtr, "y_tr":np.asarray(y_tr), "Xva":Xva, "y_va":np.asarray(y_va),
        "models":models, "cutoffs":cutoffs, "preds":preds, "strategy":chosen_strategy,
        "F1":f1, "Precision":prc, "Recall":rcl, "AUC":auc, "cat_cols_idx": cat_cols_idx,
        "best_params": {"lgbm": best_lgb, "xgb": best_xgb, "cat": best_cat}
    }

def plot_lgbm_error_trajectory(Xtr, y_tr, Xva, y_va, best_lgb_params: dict, n_estimators_big: int = 6000):
    params = LGBM_FIXED.copy()
    params.update(best_lgb_params)
    params["n_estimators"] = int(n_estimators_big)  # 길게 줘서 궤적 확인

    model = lgb.LGBMClassifier(**params)
    model.fit(
        Xtr, y_tr,
        eval_set=[(Xva, y_va)],
        eval_metric="auc",
        callbacks=[lgb.log_evaluation(0)],   # 전 구간 기록, 조기종료 X
    )
    ev = model.evals_result_
    key = "valid_0" if "valid_0" in ev else "validation_0"
    auc_list = ev[key]["auc"]
    iters = np.arange(1, len(auc_list)+1)
    val_error = 1.0 - np.array(auc_list, dtype=float)

    plt.figure(figsize=(7,4.2))
    plt.plot(iters, val_error, linewidth=1.5)
    md = params.get("max_depth", "NA"); cbt = params.get("colsample_bytree", "NA")
    plt.title(f"LGBM Error Trajectory | max_depth={md}, colsample={cbt}, lr={params.get('learning_rate')}")
    plt.xlabel("n_estimators (boosting rounds)")
    plt.ylabel("Validation Error (1 - AUC)")
    plt.grid(True, linewidth=0.3)
    plt.savefig(RESULTS_DIR / f"lgbm_error_trajectory_{SEED}.png") # ✅ 파일로 저장하는 코드 추가
    plt.close() # ✅ 메모리 해제를 위해 추가

    best_idx = int(np.argmin(val_error))
    logging.info(f"🔎 최소 에러 지점: iter={best_idx+1:,} | 1-AUC={val_error[best_idx]:.6f} | AUC={1-val_error[best_idx]:.6f}")

def plot_xgb_error_trajectory(Xtr, y_tr, Xva, y_va, best_xgb_params: dict, n_estimators_big: int = 6000):
    """
    best_xgb_params: stage1 튜닝 결과(models['xgb'].params)에서 꺼낸 dict 사용 권장
      - 사용되는 키: max_depth, colsample_bytree, subsample, learning_rate, reg_alpha, reg_lambda, scale_pos_weight(옵션)
    n_estimators_big: 궤적을 볼 만큼 크게 줌 (조기종료 없음)
    """
    # 학습/검증 DMatrix
    dtr = xgb.DMatrix(Xtr, label=y_tr)
    dva = xgb.DMatrix(Xva, label=y_va)

    # 파라미터 구성 (AUC 기준으로 에러곡선 = 1 - AUC)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_bin": 256,
        "random_state": best_xgb_params.get("random_state", 2025),
        "nthread": best_xgb_params.get("n_jobs", max(1, (os.cpu_count() or 8)//4)),
        "learning_rate": best_xgb_params.get("learning_rate", 0.05),
        "max_depth": best_xgb_params.get("max_depth", 8),
        "subsample": best_xgb_params.get("subsample", 0.1),
        "colsample_bytree": best_xgb_params.get("colsample_bytree", 0.3),
        "reg_alpha": best_xgb_params.get("reg_alpha", 0.1),
        "reg_lambda": best_xgb_params.get("reg_lambda", 0.1),
    }
    if "scale_pos_weight" in best_xgb_params:
        params["scale_pos_weight"] = best_xgb_params["scale_pos_weight"]

    evals_result = {}
    booster = xgb.train(
        params,
        dtr,
        num_boost_round=int(n_estimators_big),
        evals=[(dtr, "train"), (dva, "valid")],
        evals_result=evals_result,
        verbose_eval=False,     # 전체 궤적을 조용히 수집
    )

    auc_list = evals_result["valid"]["auc"]
    iters = np.arange(1, len(auc_list) + 1)
    val_error = 1.0 - np.array(auc_list, dtype=float)

    plt.figure(figsize=(7, 4.2))
    plt.plot(iters, val_error, linewidth=1.5)
    plt.title(f"XGBoost Error Trajectory | max_depth={params['max_depth']}, colsample={params['colsample_bytree']}, lr={params['learning_rate']}")
    plt.xlabel("n_estimators (boosting rounds)")
    plt.ylabel("Validation Error (1 - AUC)")
    plt.grid(True, linewidth=0.3)
    plt.savefig(RESULTS_DIR / f"xgb_error_trajectory_{SEED}.png") # 파일 이름은 함수에 맞게 변경
    plt.close()


    best_idx = int(np.argmin(val_error))
    logging.info(f"🔎 [XGB] 최소 에러 지점: iter={best_idx+1:,} | 1-AUC={val_error[best_idx]:.6f} | AUC={1 - val_error[best_idx]:.6f}")

def plot_cat_error_trajectory(Xtr, y_tr, Xva, y_va, cat_cols_idx, best_cat_params: dict, n_estimators_big: int = 6000):
    """
    best_cat_params: stage1 튜닝 결과(models['cat'].get_params())에서 꺼낸 dict 사용 권장
      - 사용되는 키: depth, learning_rate(고정 0.05), class_weights(있을 수 있음)
    n_estimators_big: 궤적을 볼 만큼 크게 줌 (조기종료 없음)
    """
    # Pool (범주형 인덱스 반영)
    pool_tr = Pool(Xtr, y_tr, cat_features=cat_cols_idx or None)
    pool_va = Pool(Xva, y_va, cat_features=cat_cols_idx or None)

    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",           # 궤적을 AUC 기준으로 받음
        "depth": best_cat_params.get("depth", 8),
        "learning_rate": best_cat_params.get("learning_rate", 0.05),
        "iterations": int(n_estimators_big),
        "random_seed": best_cat_params.get("random_seed", 2025),
        "verbose": False,
    }
    if "class_weights" in best_cat_params:
        params["class_weights"] = best_cat_params["class_weights"]

    model = CatBoostClassifier(**params)
    model.fit(pool_tr, eval_set=pool_va, use_best_model=False, verbose=False)
    ev = model.get_evals_result()
    # CatBoost는 키가 'validation' 또는 'learn'으로 잡힘 (버전에 따라 'Validation'일 수도 있어 보정)
    key_candidates = ["validation", "Validation", "valid"]
    mkey = next((k for k in key_candidates if k in ev), None)
    if mkey is None:
        raise RuntimeError(f"CatBoost evals_result에 validation 키가 없습니다: keys={list(ev.keys())}")

    auc_list = ev[mkey]["AUC"]
    iters = np.arange(1, len(auc_list)+1)
    val_error = 1.0 - np.array(auc_list, dtype=float)

    plt.figure(figsize=(7, 4.2))
    plt.plot(iters, val_error, linewidth=1.5)
    cw = best_cat_params.get("class_weights", None)
    cw_str = f", cw={cw}" if cw is not None else ""
    plt.title(f"CatBoost Error Trajectory | depth={params['depth']}, lr={params['learning_rate']}{cw_str}")
    plt.xlabel("iterations")
    plt.ylabel("Validation Error (1 - AUC)")
    plt.grid(True, linewidth=0.3)
    plt.savefig(RESULTS_DIR / f"cat_error_trajectory_{SEED}.png")
    plt.close()

    best_idx = int(np.argmin(val_error))
    logging.info(f"🔎 [CAT] 최소 에러 지점: iter={best_idx+1:,} | 1-AUC={val_error[best_idx]:.6f} | AUC={1 - val_error[best_idx]:.6f}")

def run_stage1_for_seeds(seeds=SEEDS, trials=50, do_plots=False, is_test_mode=False):
    results = []

    # ✅ 테스트 모드일 경우 Optuna trial 횟수 조정
    if is_test_mode:
        logging.info("🔥 Test mode: Optuna trials reduced to 3.")
        trials = 3

    # --- ✅ 추가된 부분: 이미 완료된 시드 건너뛰기 ---
    completed_seeds = set()
    for f in RESULTS_DIR.glob("seed_*.joblib"):
        try:
            # 파일명에서 시드 번호 추출 (예: seed_2021.joblib -> 2021)
            completed_seeds.add(int(f.stem.split('_')[1]))
        except (ValueError, IndexError):
            continue

    logging.info(f"▶️ Found {len(completed_seeds)} completed seeds: {sorted(list(completed_seeds))}")

    for sd in seeds:
        logging.info("="*70)

        # --- ✅ 추가된 부분: 시드 실행 여부 확인 ---
        if sd in completed_seeds:
            logging.info(f"▶▶ Skipping seed {sd} (already completed). Loading from file.")
            res = joblib.load(RESULTS_DIR / f"seed_{sd}.joblib")
            results.append(res)
            continue
        # ---------------------------------------------

        logging.info(f"▶▶ Stage1 run for seed {sd}")
        res = run_stage1_only(trials=trials, seed=sd, is_test_mode=is_test_mode)

        # --- ✅ 추가된 부분: 현재 시드 결과 저장 ---
        # 모델 객체 때문에 용량이 클 수 있으므로, 필요한 정보만 저장할 수도 있습니다.
        # 여기서는 일단 전체를 저장합니다.
        save_path = RESULTS_DIR / f"seed_{sd}.joblib"
        joblib.dump(res, save_path)
        logging.info(f"✅ Saved results for seed {sd} to: {save_path}")
        # ---------------------------------------------

        results.append(res)

        if do_plots:
            best_lgb = res["models"]["lgbm"].get_params()
            plot_lgbm_error_trajectory(res["Xtr"], res["y_tr"], res["Xva"], res["y_va"], best_lgb, n_estimators_big=6000)
            best_xgb = res["models"]["xgb"].params
            plot_xgb_error_trajectory(res["Xtr"], res["y_tr"], res["Xva"], res["y_va"], best_xgb, n_estimators_big=6000)
            best_cat = res["models"]["cat"].get_params()
            plot_cat_error_trajectory(res["Xtr"], res["y_tr"], res["Xva"], res["y_va"], res["cat_cols_idx"], best_cat_params=best_cat, n_estimators_big=6000)

    # 요약표
    summary_rows = []
    for res in results:
        best_params = res.get("best_params", {})
        row = {
            "seed":      int(res["seed"]),
            "F1":        float(res["F1"]),
            "Precision": float(res["Precision"]),
            "Recall":    float(res["Recall"]),
            "AUC":       float(res.get("AUC", float("nan"))),
            "strategy":  str(res["strategy"]),
            "best_lgbm": json.dumps(best_params.get("lgbm", {}), ensure_ascii=False),
            "best_xgb":  json.dumps(best_params.get("xgb",  {}), ensure_ascii=False),
            "best_cat":  json.dumps(best_params.get("cat",  {}), ensure_ascii=False),
            "cut_stage1": json.dumps(res.get("cutoffs", {}), ensure_ascii=False),
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("seed")

    logging.info("\n===== Stage1 multi-seed summary (with AUC & Best Params) =====")
    logging.info(summary.to_string(index=False))

    summary_csv_path = RESULTS_DIR / "stage1_summary_results.csv"
    summary.to_csv(summary_csv_path, index=False)
    logging.info(f"✅ Summary results (with AUC & Best Params) saved to: {summary_csv_path}")

    return {"by_seed": results, "summary": summary}


# =====================================================================================
# ---- 4. SCRIPT EXECUTION LOGIC
# =====================================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run the LTV prediction pipeline.")
    parser.add_argument(
        '--stage',
        type=str,
        default='stage1',
        choices=['stage1', 'all'],
        help="Which part of the pipeline to run: 'stage1' for intermediate results, 'all' for the full pipeline."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help=f"A specific random seed to run. Defaults to {DEFAULT_SEED}."
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help="Number of Optuna trials for Stage 1 hyperparameter tuning."
    )
    # ✅ 테스트 모드 인자 추가
    parser.add_argument(
        '--test_mode',
        action='store_true',  # 이 옵션을 쓰면 True가 됨
        help="Run in test mode with sampled data and fewer trials for quick debugging."
    )

    # 먼저 파싱
    args = parser.parse_args()

    # 그 다음에 test_mode 분기
    if args.test_mode:
        logging.info("⚡ Test mode: Narrowing search spaces and trials.")
        OPTUNA_TRIALS.update({"stage1": 3})
        # Stage1용 범위 축소
        global LGBM_N_EST_RANGE, XGB_N_EST_RANGE, CAT_ITER_RANGE
        LGBM_N_EST_RANGE = (50, 100)
        XGB_N_EST_RANGE  = (50, 100)
        CAT_ITER_RANGE   = (50, 100)

    # ==================================================================
    # ✅ 로거 설정 (수정됨: 에러 핸들링 추가)
    # ==================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    log_file_path = f'ltv_pipeline_{timestamp}_{unique_id}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout) # 터미널 표준 출력으로 변경
        ]
    )

    # ✅ 모든 예외(Exception)를 로그 파일에 기록하도록 설정
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.error("💥 Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    # =================================================================

    # 이제 print 대신 logging.info를 사용합니다.
    logging.info("=" * 80)
    logging.info(f"🚀 Starting LTV Prediction Pipeline")
    logging.info(f"🔹 Execution Stage: {args.stage}")
    logging.info(f"🔹 Random Seed: {args.seed}")
    logging.info(f"🔹 Data Directory: {DATA_DIR.resolve()}")
    logging.info(f"🔹 Device: {DEVICE}")
    logging.info("=" * 80)

    if not DATA_DIR.exists():
        logging.info(f"❌ FATAL: Data directory not found at {DATA_DIR}")
        logging.info("Please ensure your data is structured correctly under /data.")
        return

    # Set up environment
    pd.options.display.float_format = '{:,.2f}'.format
    warnings.filterwarnings("ignore")
    CPU = os.cpu_count() or 2
    os.environ["OMP_NUM_THREADS"] = str(CPU)
    
    try:
        if args.stage == 'stage1':
            # ✅ run_stage1_for_seeds 함수만 호출
            run_stage1_for_seeds(seeds=SEEDS, trials=args.trials, is_test_mode=args.test_mode, do_plots=True)
        
        elif args.stage == 'all': # ✅ elif를 try 안으로 이동
            logging.info("\n▶️ Running the full pipeline across all seeds...")
            # agg_results = run_all_seeds(seeds=SEEDS, is_test_mode=args.test_mode) # test_mode 전달 추가
            # final_evaluation = evaluate_after_seeds(agg_results)
            # logging.info("\n✅ Full pipeline finished.")
            # logging.info(f"Final evaluation metrics: {final_evaluation['ensemble_mean']['metrics']}")
            logging.info("NOTE: 'all' stages execution logic needs to be fully implemented based on notebook.")
            
    except Exception as e: # ✅ try 블록이 끝난 직후에 except가 오도록 수정
        logging.error(f"☠️ A critical error occurred during pipeline execution.")
        logging.error(traceback.format_exc()) # 에러 상세 내용 기록

if __name__ == "__main__":
    main()