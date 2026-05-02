import 'package:cloud_firestore/cloud_firestore.dart';

class CropRecommendationService {
  static const String _collection = "croprecomdation";

  // Convert prediction text to your Firestore doc id
  // Example: "Rice" -> "RICE", "Red Onion" -> "ONION"
  static String toDocId(String prediction) {
    final p = prediction.trim().toUpperCase();

    // handle common variations from ML output
    if (p.contains("PADDY") || p == "RICE") return "RICE";
    if (p.contains("MAIZE") || p == "CORN") return "MAIZE";
    if (p.contains("ONION")) return "ONION";
    if (p.contains("BEET")) return "BEETROOT";
    if (p.contains("RADISH")) return "RADISH";
    if (p.contains("CHICK")) return "CHICKPEA";

    // fallback: use uppercase word
    return p.replaceAll(RegExp(r"\s+"), "_");
  }

  static Future<Map<String, dynamic>?> fetchByPrediction(
      String prediction) async {
    final docId = toDocId(prediction);

    final snap = await FirebaseFirestore.instance
        .collection(_collection)
        .doc(docId)
        .get();

    if (!snap.exists) return null;

    return {
      "docId": docId,
      ...?snap.data(),
    };
  }
}
