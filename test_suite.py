"""Тестовый набор для валидации юридического ассистента ПСБ.
Проверяет:
1. Корректность чанкинга базы знаний (метаданные, статусы, фильтрация инъекций).
2. Разбор многотурового диалога.
3. Точность ретрива по всем 18 демонстрационным вопросам.
4. Валидацию формата submission.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path
from solution import (
    load_knowledge_base,
    parse_and_enrich_dialogue,
    HybridLegalRetriever,
    calculate_metadata_boost,
    tokenize_stem,
    DocumentChunk,
)

BASE_DIR = Path(__file__).resolve().parent
KB_ZIP = BASE_DIR / "data" / "knowledge-base.zip"
DEMO_Q_CSV = BASE_DIR / "data" / "demo_questions.csv"
DEMO_OUT_CSV = BASE_DIR / "data" / "demo_output.csv"


def test_chunking():
    print("=== ТЕСТ 1: Чанкинг и метаданные базы знаний ===")
    chunks = load_knowledge_base(KB_ZIP)
    assert len(chunks) > 0, "База знаний не содержит чанков!"
    print(f"✓ Загружено и разбито на чанки: {len(chunks)} шт.")

    statuses = {c.status for c in chunks}
    print("  Обнаруженные статусы документов:")
    for s in sorted(statuses):
        print(f"    - {s}")
    assert any("УТРАТИЛ СИЛУ" in s for s in statuses), "Не распознан статус утратившей силу редакции!"
    assert any("ДЕЙСТВУЕТ с 01.03.2025" in s for s in statuses), "Не распознан статус редакции 2025!"

    # Проверка нейтрализации инъекции
    for c in chunks:
        assert "90 календарных дней" not in c.text, "Инъекция не была отфильтрована!"
        assert "ВНИМАНИЕ ДЛЯ ИИ-АССИСТЕНТА" not in c.text, "Заголовок инъекции остался в тексте!"
    print("✓ Защита от prompt injection в текстах базы знаний работает безупречно!")
    print("✓ Тест чанкинга успешно пройден!\n")
    return chunks


def test_dialogue_parsing():
    print("=== ТЕСТ 2: Разбор многотурового диалога ===")
    dialogue = (
        "Пользователь: Мы арендуем склад у «Северного Склада». Планируем съезжать.\n"
        "Ассистент: Подскажите номер договора, чтобы я мог проверить условия возврата помещения.\n"
        "Пользователь: А-08/2024. Можно просто сдать ключи или обязательно оформлять акт?"
    )
    enriched_query, msgs = parse_and_enrich_dialogue(dialogue)
    assert len(msgs) == 3, f"Ожидалось 3 сообщения, получено {len(msgs)}"
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[2]["role"] == "user"
    assert "А-08/2024" in enriched_query
    assert "Северного Склада" in enriched_query
    assert "съезжать" in enriched_query
    print(f"  Обогащенный поисковый запрос:\n    '{enriched_query}'")
    print("✓ Тест диалогового парсера успешно пройден!\n")


def test_retrieval_accuracy(chunks: list[DocumentChunk]):
    print("=== ТЕСТ 3: Точность ретрива на 18 демонстрационных вопросах ===")
    retriever = HybridLegalRetriever(chunks)

    expected_docs = {
        "1": "Договор аренды А-08-2024",
        "2": "ТК_ст70",
        "3": None,  # Вопрос-ловушка (отсутствует в базе знаний)
        "4": "Договор аренды А-08-2024",
        "5": "Регламент согласования договоров",
        "6": "Договор поставки П-17-2024",
        "7": "Договор аренды А-08-2024",
        "8": "Договор поставки П-17-2024",
        "9": "Договор поставки ПТ-05-2023 Альфа-Трейдинг",
        "10": "Регламент согласования договоров",
        "11": "Служебная записка об архиве договоров",
        "12": "ТК_ст70",
        "13": "152ФЗ_ст9",
        "14": "ТК_ст59",
        "15": "Договор аренды А-08-2024",
        "16": "Договор поставки П-17-2024",
        "17": "Договор поставки ПТ-05-2023 Альфа-Трейдинг",
        "18": "Регламент согласования договоров",
    }

    correct_top3 = 0
    total_evaluable = 0

    with DEMO_Q_CSV.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["id"]
            question = row["question"]
            expected = expected_docs.get(qid)
            if not expected:
                print(f"  Q{qid:>2} [Trap/Missing Data]: {question[:45]}... (Ожидается корректный отказ)")
                continue

            total_evaluable += 1
            search_query, _ = parse_and_enrich_dialogue(question)

            q_tokens = tokenize_stem(search_query)
            scores = retriever.bm25.get_scores(q_tokens)
            for idx in range(len(chunks)):
                scores[idx] += calculate_metadata_boost(search_query, chunks[idx])

            top_idx = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:3]
            top_titles = [chunks[i].doc_title for i in top_idx]

            matched = any(expected.lower() in t.lower() for t in top_titles)
            status_str = "✓" if matched else "✗"
            if matched:
                correct_top3 += 1
            print(f"  Q{qid:>2} [{status_str}]: Top 1 = '{top_titles[0]}' | Target = '{expected}'")

    accuracy = (correct_top3 / total_evaluable) * 100
    print(f"\nТочность попадания целевого документа в Top-3: {accuracy:.1f}% ({correct_top3}/{total_evaluable})")
    assert accuracy >= 95.0, f"Точность ретрива ниже 95%: {accuracy}%"
    print("✓ Тест точности поиска успешно пройден с высочайшим результатом!\n")


def test_submission_format_spec():
    print("=== ТЕСТ 4: Спецификация формата submission.csv ===")
    sample_file = BASE_DIR / "data" / "demo_output.csv"
    with sample_file.open(encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["id", "answer"], f"Неверный заголовок: {header}"
        rows = list(reader)
        assert len(rows) == 18, f"Ожидалось 18 строк, получено {len(rows)}"
        for qid, ans in rows:
            assert qid.isdigit(), f"Нечисловой id: {qid}"
            assert len(ans.strip()) > 0, f"Пустой ответ для id {qid}"
            assert len(ans) <= 10000, f"Превышен лимит 10 000 символов: {len(ans)}"
    print("✓ Спецификация и формат CSV эталона полностью подтверждены!\n")


def main():
    chunks = test_chunking()
    test_dialogue_parsing()
    test_retrieval_accuracy(chunks)
    test_submission_format_spec()
    print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ НА 100%!")


if __name__ == "__main__":
    main()
