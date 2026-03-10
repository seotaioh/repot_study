"""
사용자 인증 및 권한 관리 모듈
- JSON 기반 사용자 설정 관리
- 역할별 열람 권한 (view_scope) 제어
- 새 인원 추가/수정/삭제 API 지원
"""
import json
from pathlib import Path
from config import BASE_DIR

USERS_CONFIG_PATH = BASE_DIR / "data" / "users_config.json"


def load_users_config() -> dict:
    """사용자 설정 파일 로드"""
    if USERS_CONFIG_PATH.exists():
        with open(USERS_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"users": {}, "roles_hierarchy": {}}


def save_users_config(config: dict):
    """사용자 설정 파일 저장"""
    with open(USERS_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def get_user(name: str) -> dict | None:
    """사용자 정보 조회 (없으면 None)"""
    config = load_users_config()
    user_data = config.get("users", {}).get(name)
    if user_data:
        return {"name": name, **user_data}
    return None


def get_all_users() -> dict:
    """전체 사용자 목록 반환"""
    config = load_users_config()
    return config.get("users", {})


def add_user(name: str, role: str, department: str = "연구소",
             view_scope="self", can_delete: bool = False) -> dict:
    """새 사용자 추가"""
    config = load_users_config()
    if name in config.get("users", {}):
        return {"success": False, "message": f"이미 존재하는 사용자입니다: {name}"}

    config.setdefault("users", {})[name] = {
        "role": role,
        "department": department,
        "view_scope": view_scope,
        "can_delete": can_delete,
    }
    save_users_config(config)
    return {"success": True, "message": f"사용자 추가 완료: {name} ({role})"}


def update_user(name: str, updates: dict) -> dict:
    """사용자 정보 수정"""
    config = load_users_config()
    if name not in config.get("users", {}):
        return {"success": False, "message": f"존재하지 않는 사용자입니다: {name}"}

    allowed_fields = {"role", "department", "view_scope", "can_delete"}
    for key, value in updates.items():
        if key in allowed_fields:
            config["users"][name][key] = value
    save_users_config(config)
    return {"success": True, "message": f"사용자 정보 수정 완료: {name}"}


def delete_user(name: str) -> dict:
    """사용자 삭제"""
    config = load_users_config()
    if name not in config.get("users", {}):
        return {"success": False, "message": f"존재하지 않는 사용자입니다: {name}"}

    del config["users"][name]
    save_users_config(config)
    return {"success": True, "message": f"사용자 삭제 완료: {name}"}


def check_view_permission(current_user: str, target_researcher: str) -> bool:
    """
    현재 사용자가 대상 연구원의 보고서를 열람할 수 있는지 확인

    view_scope 규칙:
    - "all": 전체 열람 가능
    - "self": 본인 보고서만 열람 가능
    - ["이름1", "이름2", ...]: 지정된 연구원만 열람 가능
    """
    user = get_user(current_user)
    if not user:
        return False

    scope = user.get("view_scope", "self")

    if scope == "all":
        return True

    if scope == "self":
        return current_user == target_researcher

    # 리스트인 경우: 지정된 연구원 + 본인 열람 가능
    if isinstance(scope, list):
        return target_researcher in scope or current_user == target_researcher

    return False


def get_viewable_researchers(current_user: str) -> list | None:
    """
    현재 사용자가 열람 가능한 연구원 목록 반환
    - None: 전체 열람 가능 (필터 불필요)
    - []: 본인만 열람 가능
    - ["이름1", ...]: 지정 연구원 열람 가능
    """
    user = get_user(current_user)
    if not user:
        return []

    scope = user.get("view_scope", "self")

    if scope == "all":
        return None  # 전체 열람 → 필터 없음

    if scope == "self":
        return [current_user]

    if isinstance(scope, list):
        # 지정 목록 + 본인
        viewable = list(scope)
        if current_user not in viewable:
            viewable.append(current_user)
        return viewable

    return [current_user]


def can_delete_file(current_user: str) -> bool:
    """파일 삭제 권한 확인"""
    user = get_user(current_user)
    if not user:
        return False
    return user.get("can_delete", False)
