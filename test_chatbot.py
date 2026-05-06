"""Quick test script for chatbot classification accuracy - outputs JSON."""
import requests
import json

tests = [
    ("I am happy and feeling great today", "Normal"),
    ("I feel completely hopeless, like nothing matters anymore", "Depression"),
    ("I feel anxious all the time, my heart is racing and I cant breathe", "Anxiety"),
    ("I hear voices that arent real, people are watching me", "Schizophrenia"),
    ("My moods swing from extreme highs to extreme lows, bipolar", "Bipolar"),
    ("I fear everyone will abandon me, my emotions are out of control", "BPD"),
    ("I am struggling with my mental health and need help", "Mental Illness"),
    ("Life is wonderful, I am grateful for everything", "Normal"),
    ("I want to end it all, I cant go on", "Crisis"),
    ("Today was a great day, I went for a walk and felt peaceful", "Normal"),
    ("I feel depressed and worthless", "Depression"),
    ("I keep having panic attacks and cant stop worrying", "Anxiety"),
]

results = []
passed = 0
failed = 0

for msg, expected in tests:
    try:
        r = requests.post("http://localhost:8000/api/chatbot", json={"message": msg})
        data = r.json()
        assess = data.get("mental_health_assessment", {})
        detected = assess.get("detected_condition", "N/A")
        conf = assess.get("confidence", 0)
        ml_pred = assess.get("ml_prediction", "N/A")
        is_normal = assess.get("is_normal", False)
        
        match = detected == expected
        if match:
            passed += 1
        else:
            failed += 1
        
        results.append({
            "input": msg[:60],
            "expected": expected,
            "detected": detected,
            "match": match,
            "confidence": round(conf, 3),
            "ml_prediction": ml_pred,
            "is_normal": is_normal,
        })
    except Exception as e:
        failed += 1
        results.append({"input": msg[:60], "expected": expected, "error": str(e)})

output = {
    "total": len(tests),
    "passed": passed,
    "failed": failed,
    "results": results,
}

with open("test_output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# Also print summary
for r in results:
    status = "PASS" if r.get("match") else "FAIL"
    print(f"[{status}] {r['expected']:18s} -> {r.get('detected', 'ERR'):18s} | {r['input'][:50]}")
print(f"TOTAL: {passed}/{passed+failed}")
