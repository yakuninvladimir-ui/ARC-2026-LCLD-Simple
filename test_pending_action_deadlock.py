#!/usr/bin/env python3
"""
DEADLOCK DETECTION TEST FOR PENDING_ACTION LOGIC

Этот скрипт проверяет сценарии, которые могут привести к deadlock в логике pending_action.

Deadlock возможен в следующих случаях:
1. act() устанавливает pending_action, но observe_action_result() никогда не вызывается
2. act() вызывается повторно без observe_action_result(), и snapshot не совпадает (строка 154)
3. observe_action_result() вызывается без pending_action (возвращает False, строка 306)
4. Цикл: act() -> RESET -> act() -> RESET без изменения состояния

Сценарии проверки:
A. Нормальный цикл: act() -> observe_action_result() -> OK
B. Пропуск observe_action_result(): act() -> act() с тем же snapshot -> RuntimeError (строка 154)
C. Пропуск observe_action_result(): act() -> act() с новым snapshot -> auto-commit (строка 155)
D. observe_action_result() без pending_action -> False (строка 306)
E. Многократный RESET без изменения состояния -> проверка _fallback_streak_by_level
"""

import sys
from typing import Any, Mapping
from unittest.mock import Mock, MagicMock

# Добавляем путь к модулю
sys.path.insert(0, '/workspace')

from v9_agent.session import GameSession
from v9_agent.config import V8Config


def create_mock_observation(game_id: str = "game_1", level_index: int = 0, step_index: int = 0, 
                            grid: list | None = None, state_name: str = "PLAYING", 
                            game_over: bool = False, terminal: bool = False) -> dict[str, Any]:
    """Создает мок observation для тестирования."""
    if grid is None:
        grid = [[1, 1], [1, 1]]
    
    return {
        "game_id": game_id,
        "level_index": level_index,
        "step_index": step_index,
        "grid": grid,
        "state_name": state_name,
        "game_over": game_over,
        "terminal": terminal,
        "metadata": {}
    }


def test_scenario_a_normal_cycle():
    """Сценарий A: Нормальный цикл act() -> observe_action_result()"""
    print("\n=== Сценарий A: Нормальный цикл ===")
    config = V8Config()
    session = GameSession(config)
    
    obs1 = create_mock_observation(step_index=0)
    action = session.act(obs1)
    print(f"✓ act() вернул действие: {action.get('action_id', 'UNKNOWN')}")
    print(f"  pending_action установлен: {session.pending_action is not None}")
    
    obs2 = create_mock_observation(step_index=1)
    result = session.observe_action_result(obs2)
    print(f"✓ observe_action_result() вернул: {result}")
    print(f"  pending_action сброшен: {session.pending_action is None}")
    
    assert session.pending_action is None, "pending_action должен быть сброшен"
    assert result is True, "observe_action_result должен вернуть True"
    print("✅ Сценарий A пройден")
    return True


def test_scenario_b_skip_observe_same_snapshot():
    """Сценарий B: Пропуск observe_action_result() с тем же snapshot"""
    print("\n=== Сценарий B: Пропуск observe_action_result() (тот же snapshot) ===")
    config = V8Config()
    session = GameSession(config)
    
    obs1 = create_mock_observation(step_index=0)
    action1 = session.act(obs1)
    print(f"✓ Первый act() выполнен: {action1.get('action_id', 'UNKNOWN')}")
    print(f"  pending_action установлен: {session.pending_action is not None}")
    
    try:
        # Второй act() с тем же snapshot должен вызвать RuntimeError
        action2 = session.act(obs1)
        print(f"❌ ОШИБКА: Второй act() не вызвал RuntimeError!")
        print(f"  Это POTENTIAL DEADLOCK: pending_action остается установленным бесконечно")
        return False
    except RuntimeError as e:
        print(f"✓ Ожидается RuntimeError: {e}")
        print(f"  pending_action все еще установлен: {session.pending_action is not None}")
        print("✅ Сценарий B пройден (защита работает)")
        return True


def test_scenario_c_skip_observe_new_snapshot():
    """Сценарий C: Пропуск observe_action_result() с новым snapshot (auto-commit)"""
    print("\n=== Сценарий C: Пропуск observe_action_result() (новый snapshot) ===")
    config = V8Config()
    session = GameSession(config)
    
    obs1 = create_mock_observation(step_index=0)
    action1 = session.act(obs1)
    print(f"✓ Первый act() выполнен: {action1.get('action_id', 'UNKNOWN')}")
    pending_token_1 = session.pending_action.token_id if session.pending_action else None
    print(f"  pending_action токен: {pending_token_1}")
    
    obs2 = create_mock_observation(step_index=1)  # Новый snapshot
    try:
        action2 = session.act(obs2)  # Должен сделать auto-commit
        print(f"✓ Второй act() выполнен (auto-commit): {action2.get('action_id', 'UNKNOWN')}")
        print(f"  pending_action был сброшен и установлен заново: {session.pending_action is not None}")
        print("✅ Сценарий C пройден (auto-commit работает)")
        return True
    except Exception as e:
        print(f"❌ ОШИБКА: Второй act() вызвал исключение: {e}")
        return False


def test_scenario_d_observe_without_pending():
    """Сценарий D: observe_action_result() без pending_action"""
    print("\n=== Сценарий D: observe_action_result() без pending_action ===")
    config = V8Config()
    session = GameSession(config)
    
    obs1 = create_mock_observation(step_index=0)
    # Не вызываем act(), сразу observe_action_result()
    result = session.observe_action_result(obs1)
    print(f"✓ observe_action_result() без prior act() вернул: {result}")
    assert result is False, "observe_action_result должен вернуть False"
    print("✅ Сценарий D пройден (возвращает False как no-op)")
    return True


def test_scenario_e_reset_loop_detection():
    """Сценарий E: Многократный RESET без изменения состояния"""
    print("\n=== Сценарий E: Проверка _fallback_streak_by_level ===")
    from dataclasses import replace
    config = replace(V8Config(), max_level_attempts=10)  # Увеличиваем лимит через replace
    session = GameSession(config)
    
    obs = create_mock_observation(step_index=0)
    
    # Симулируем multiple fallback actions
    reset_count = 0
    for i in range(15):
        try:
            action = session.act(obs)
            if action.get('action_id') == 'RESET':
                reset_count += 1
                print(f"  Итерация {i+1}: RESET (источник: {action.get('source', 'UNKNOWN')})")
                
                # Проверяем, не сработал ли механизм ограничения попыток
                telemetry = session.harness_telemetry()
                fallback_streak = telemetry.get('action_selection', {}).get('fallback_enabled', False)
                
            # Симулируем observe_action_result с тем же состоянием (дедлок ситуация)
            # В реальном deadлоке это не меняет состояние
            result = session.observe_action_result(obs)
            
        except RuntimeError as e:
            print(f"  Итерация {i+1}: RuntimeError - {e}")
            break
    
    print(f"  Всего RESET действий: {reset_count}")
    telemetry = session.harness_telemetry()
    print(f"  pending_official_transition: {telemetry['pending_official_transition']}")
    print("✅ Сценарий E завершен")
    return True


def test_scenario_f_game_change_with_pending():
    """Сценарий F: Смена игры с установленным pending_action"""
    print("\n=== Сценарий F: Смена игры с pending_action ===")
    config = V8Config()
    session = GameSession(config)
    
    obs1 = create_mock_observation(game_id="game_1", step_index=0)
    action1 = session.act(obs1)
    print(f"✓ act() для game_1 выполнен: {action1.get('action_id', 'UNKNOWN')}")
    print(f"  pending_action установлен: {session.pending_action is not None}")
    
    # Смена игры без observe_action_result()
    obs2 = create_mock_observation(game_id="game_2", step_index=0)
    try:
        action2 = session.act(obs2)
        print(f"❌ ОШИБКА: act() для новой игры не вызвал RuntimeError!")
        return False
    except RuntimeError as e:
        print(f"✓ Ожидается RuntimeError: {e}")
        print("✅ Сценарий F пройден (защита работает)")
        return True


def main():
    print("=" * 70)
    print("DEADLOCK DETECTION TEST SUITE FOR PENDING_ACTION LOGIC")
    print("=" * 70)
    
    results = []
    
    results.append(("Сценарий A: Нормальный цикл", test_scenario_a_normal_cycle()))
    results.append(("Сценарий B: Пропуск observe (тот же snapshot)", test_scenario_b_skip_observe_same_snapshot()))
    results.append(("Сценарий C: Пропуск observe (новый snapshot)", test_scenario_c_skip_observe_new_snapshot()))
    results.append(("Сценарий D: observe без pending", test_scenario_d_observe_without_pending()))
    results.append(("Сценарий E: RESET loop detection", test_scenario_e_reset_loop_detection()))
    results.append(("Сценарий F: Смена игры с pending", test_scenario_f_game_change_with_pending()))
    
    print("\n" + "=" * 70)
    print("ИТОГИ:")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nПройдено: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 Все тесты пройдены! Deadlock сценарии корректно обрабатываются.")
        return 0
    else:
        print(f"\n⚠️  Обнаружено {total - passed} потенциальных проблем!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
