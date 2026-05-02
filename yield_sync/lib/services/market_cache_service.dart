import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class MarketCacheService {
  MarketCacheService._();

  static final FirebaseAuth _auth = FirebaseAuth.instance;
  static final FirebaseFirestore _db = FirebaseFirestore.instance;

  static String _uid() {
    final u = _auth.currentUser;
    if (u == null) throw Exception("Not logged in");
    return u.uid;
  }

  static CollectionReference<Map<String, dynamic>> _col() {
    return _db.collection("users").doc(_uid()).collection("market_cache");
  }

  // ------------------ AVG BY MARKET ------------------
  static Future<void> saveAvgByMarket({
    required String crop,
    required int days,
    required Map<String, dynamic> payload,
  }) async {
    final docId = "avg_${crop}_$days"; // overwrite snapshot
    await _col().doc(docId).set({
      "type": "avg_by_market",
      "crop": crop,
      "days": days,
      "createdAt": FieldValue.serverTimestamp(),
      "payload": payload,
    }, SetOptions(merge: true));
  }

  static Future<Map<String, dynamic>?> loadAvgByMarket({
    required String crop,
    required int days,
  }) async {
    final docId = "avg_${crop}_$days";
    final snap = await _col().doc(docId).get();
    if (!snap.exists) return null;
    final data = snap.data();
    return (data?["payload"] as Map?)?.cast<String, dynamic>();
  }

  // ------------------ TRENDS ------------------
  static Future<void> saveTrends({
    required String crop,
    required String market,
    required Map<String, dynamic> payload,
  }) async {
    final safeMarket = market.replaceAll("/", "-");
    final docId = "trends_${crop}_$safeMarket";
    await _col().doc(docId).set({
      "type": "trends",
      "crop": crop,
      "market": market,
      "createdAt": FieldValue.serverTimestamp(),
      "payload": payload,
    }, SetOptions(merge: true));
  }

  static Future<Map<String, dynamic>?> loadTrends({
    required String crop,
    required String market,
  }) async {
    final safeMarket = market.replaceAll("/", "-");
    final docId = "trends_${crop}_$safeMarket";
    final snap = await _col().doc(docId).get();
    if (!snap.exists) return null;
    final data = snap.data();
    return (data?["payload"] as Map?)?.cast<String, dynamic>();
  }
}
