import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class FertiliserHistoryService {
  final _db = FirebaseFirestore.instance;
  final _auth = FirebaseAuth.instance;

  Future<void> saveFertiliserPrediction({
    required Map<String, dynamic> request,
    required Map<String, dynamic> response,
  }) async {
    final user = _auth.currentUser;
    if (user == null) throw Exception("Not logged in");

    final uid = user.uid;

    await _db
        .collection("users")
        .doc(uid)
        .collection("fertiliser_predictions")
        .add({
      "uid": uid,
      "request": request,
      "response": response,
      "createdAt": FieldValue.serverTimestamp(),
    });
  }

  // ✅ NEW: save total-yield calculation
  Future<void> saveTotalYieldCalc({
    required double areaAcres,
    required double yieldKgPerAcre,
    required double totalYieldKg,
    required String source,
    required String fertiliserType,
    required String cropLabel,
    required String stageLabel,
  }) async {
    final user = _auth.currentUser;
    if (user == null) throw Exception("Not logged in");

    final uid = user.uid;

    await _db
        .collection("users")
        .doc(uid)
        .collection("fertiliser_total_yield")
        .add({
      "uid": uid,
      "area_acres": areaAcres,
      "yield_kg_per_acre": yieldKgPerAcre,
      "total_yield_kg": totalYieldKg,
      "source": source,
      "fertiliser_type": fertiliserType,
      "crop": cropLabel,
      "growth_stage": stageLabel,
      "createdAt": FieldValue.serverTimestamp(),
    });
  }
}
