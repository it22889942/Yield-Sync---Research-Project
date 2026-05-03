import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class EquipmentBookingService {
  static final _db = FirebaseFirestore.instance;
  static final _auth = FirebaseAuth.instance;

  static const String equipmentsCol = "equipments";
  static const String bookingsCol = "equipment_bookings";

  static String _fmt(DateTime d) =>
      "${d.year.toString().padLeft(4, "0")}-${d.month.toString().padLeft(2, "0")}-${d.day.toString().padLeft(2, "0")}";

  /// ✅ Find correct equipments/{docId} even if docId != equipmentId
  static Future<DocumentReference<Map<String, dynamic>>> _findEquipmentRef(
    String equipmentId,
  ) async {
    final id = equipmentId.trim();
    if (id.isEmpty) throw Exception("equipmentId is empty");

    // 1) Try docId == equipmentId
    final directRef = _db.collection(equipmentsCol).doc(id);
    final directSnap = await directRef.get();
    if (directSnap.exists) return directRef;

    // 2) Try Equipment_ID field (your dataset)
    final q1 = await _db
        .collection(equipmentsCol)
        .where("Equipment_ID", isEqualTo: id)
        .limit(1)
        .get();
    if (q1.docs.isNotEmpty) return q1.docs.first.reference;

    // 3) Try equipmentId field (some datasets)
    final q2 = await _db
        .collection(equipmentsCol)
        .where("equipmentId", isEqualTo: id)
        .limit(1)
        .get();
    if (q2.docs.isNotEmpty) return q2.docs.first.reference;

    throw Exception("Equipment not found for Equipment_ID: $id");
  }

  static Future<void> createBooking({
    required String equipmentId, // ex: "E0431"
    required DateTime startDate,
    required DateTime endDate,
    required String type, // full_day / half_day / per_hour
    String? note,
  }) async {
    final u = _auth.currentUser;
    if (u == null) throw Exception("Not logged in");

    final start = DateTime(startDate.year, startDate.month, startDate.day);
    final end = DateTime(endDate.year, endDate.month, endDate.day);
    if (end.isBefore(start)) {
      throw Exception("End date must be after start date");
    }

    // ✅ farmer profile
    final farmerDoc = await _db.collection("users").doc(u.uid).get();
    final farmer = farmerDoc.data() ?? {};
    final farmerName = (farmer["username"] ?? "").toString();
    final farmerPhone = (farmer["phone"] ?? "").toString();
    final farmerEmail = (farmer["email"] ?? "").toString();

    // ✅ equipment ref (docId mismatch safe)
    final eqRef = await _findEquipmentRef(equipmentId);
    final eqSnap = await eqRef.get();
    final eq = eqSnap.data() ?? {};

    // ✅ owner snapshot (from equipment doc always available)
    String ownerName = (eq["Equipment_Owner_Name"] ?? "Owner").toString();
    String ownerPhone = (eq["Owner_Contact_No"] ?? "—").toString();
    String ownerEmail = "—";

    // ✅ ownerUid might be missing (UNLINKED scenario)
    final ownerUid = (eq["ownerUid"] ?? "").toString().trim();

    // ✅ if ownerUid exists, try to get email/phone from users
    if (ownerUid.isNotEmpty) {
      try {
        final ownerDoc = await _db.collection("users").doc(ownerUid).get();
        final owner = ownerDoc.data() ?? {};
        ownerName = (owner["username"] ??
                owner["firstName"] ??
                owner["name"] ??
                ownerName)
            .toString();
        ownerPhone = (owner["phone"] ?? ownerPhone).toString();
        ownerEmail = (owner["email"] ?? "—").toString();
      } catch (_) {}
    }

    // ✅ IMPORTANT:
    // Firestore rules only require equipmentOwnerUid is string
    // so "" is OK (still a string) -> booking can be saved even if owner not linked
    await _db.collection(bookingsCol).add({
      "equipmentId": equipmentId.trim(),
      "equipmentDocId": eqRef.id,

      "equipmentOwnerUid": ownerUid, // can be "" (unlinked) ✅
      "equipmentOwnerName": ownerName,
      "equipmentOwnerPhone": ownerPhone,
      "equipmentOwnerEmail": ownerEmail,

      "farmerUid": u.uid,
      "farmerName": farmerName,
      "farmerPhone": farmerPhone,
      "farmerEmail": farmerEmail,

      "startDate": _fmt(start),
      "endDate": _fmt(end),
      "type": type,
      "note": (note ?? "").trim(),

      "status": "pending",
      "createdAt": FieldValue.serverTimestamp(),
      "updatedAt": FieldValue.serverTimestamp(),
    });
  }

  /// ✅ Farmer reads his equipment bookings
  static Stream<QuerySnapshot<Map<String, dynamic>>> myBookings() {
    final u = _auth.currentUser;
    if (u == null) return const Stream.empty();
    return _db
        .collection(bookingsCol)
        .where("farmerUid", isEqualTo: u.uid)
        .snapshots();
  }

  /// ✅ Backward-compatible (ProfileScreen uses updateStatus)
  static Future<void> updateStatus({
    required String bookingId,
    required String status,
  }) async {
    await _db.collection(bookingsCol).doc(bookingId).update({
      "status": status,
      "updatedAt": FieldValue.serverTimestamp(),
    });
  }

  /// ✅ Farmer cancel pending booking (clean API)
  static Future<void> cancelBooking({
    required String bookingId,
  }) async {
    await updateStatus(bookingId: bookingId, status: "cancelled");
  }

  /// ✅ Owner accept/reject/complete (owner screen)
  static Future<void> ownerUpdateStatus({
    required String bookingId,
    required String status, // accepted / rejected / completed
  }) async {
    await updateStatus(bookingId: bookingId, status: status);
  }

  /// ✅ Equipment owner one-time link (IMPORTANT!)
  /// Finds correct equipment doc and sets ownerUid = my uid
  static Future<void> linkMyEquipmentOwnerUid() async {
    final u = _auth.currentUser;
    if (u == null) throw Exception("Not logged in");

    final userDoc = await _db.collection("users").doc(u.uid).get();
    final data = userDoc.data() ?? {};

    // must be "E0431" etc (same as equipmentId field in equipments)
    final equipmentId = (data["equipmentId"] ?? "").toString().trim();
    if (equipmentId.isEmpty)
      throw Exception("Your user doc missing equipmentId.");

    final eqRef = await _findEquipmentRef(equipmentId);

    // rules allow update if isEquipmentUser && myEquipmentId == equipmentId
    await eqRef.update({
      "ownerUid": u.uid,
      "updatedAt": FieldValue.serverTimestamp(),
    });
  }
}
