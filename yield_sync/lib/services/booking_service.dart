import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class BookingService {
  static final _db = FirebaseFirestore.instance;
  static final _auth = FirebaseAuth.instance;

  static String _dateKey(DateTime d) {
    final y = d.year.toString().padLeft(4, "0");
    final m = d.month.toString().padLeft(2, "0");
    final day = d.day.toString().padLeft(2, "0");
    return "$y-$m-$day";
  }

  static List<DateTime> _daysInRange(DateTime start, DateTime end) {
    final s = DateTime(start.year, start.month, start.day);
    final e = DateTime(end.year, end.month, end.day);
    final out = <DateTime>[];

    DateTime cur = s;
    while (!cur.isAfter(e)) {
      out.add(cur);
      cur = cur.add(const Duration(days: 1));
    }
    return out;
  }

  /// ✅ Create booking request (PENDING)
  /// Rule: same labour can't be booked by multiple users on SAME DAY (full lock)
  static Future<void> createRequest({
    required String labourId,
    required DateTime startDate,
    required DateTime endDate,
    required bool halfDay, // true => half-day, false => full-day
    String? note,
  }) async {
    final u = _auth.currentUser;
    if (u == null) throw Exception("Not logged in");

    final start = DateTime(startDate.year, startDate.month, startDate.day);
    final end = DateTime(endDate.year, endDate.month, endDate.day);

    if (end.isBefore(start)) {
      throw Exception("End date must be after start date");
    }

    final farmerUid = u.uid;

    // ✅ load farmer profile (store contact details inside booking doc)
    final me = await _db.collection("users").doc(farmerUid).get();
    final meData = me.data() ?? {};
    final farmerName = (meData["username"] ?? "").toString();
    final farmerPhone = (meData["phone"] ?? "").toString();
    final farmerEmail = (meData["email"] ?? "").toString();

    // Prefer owner mapping from labour post itself (new posting flow)
    String? labourUid;
    Map<String, dynamic>? labourData;

    final labourDoc = await _db.collection("labours").doc(labourId).get();
    if (labourDoc.exists) {
      labourData = labourDoc.data();
    } else {
      // Backward compatibility when Labour_ID is not used as Firestore doc id.
      final labourByField = await _db
          .collection("labours")
          .where("Labour_ID", isEqualTo: labourId)
          .limit(1)
          .get();
      if (labourByField.docs.isNotEmpty) {
        labourData = labourByField.docs.first.data();
      }
    }

    final fromPost = (labourData?["createdByUid"] ?? labourData?["ownerUid"])
        ?.toString()
        .trim();
    if (fromPost != null && fromPost.isNotEmpty) {
      labourUid = fromPost;
    }

    // Legacy fallback: look up classic labour account mapping users.labourId
    if (labourUid == null || labourUid.isEmpty) {
      final labourQ = await _db
          .collection("users")
          .where("userType", isEqualTo: "labour")
          .where("labourId", isEqualTo: labourId)
          .limit(1)
          .get();
      if (labourQ.docs.isNotEmpty) {
        labourUid = labourQ.docs.first.id;
      }
    }

    if (labourUid == null || labourUid.isEmpty) {
      throw Exception("This labour has no linked account (createdByUid/labourUid missing).");
    }

    final days = _daysInRange(start, end);

    await _db.runTransaction((tx) async {
      // 1) Firestore rule: do ALL reads before any writes.
      final keyRefs = <DocumentReference<Map<String, dynamic>>>[];
      for (final d in days) {
        final keyId = "${labourId}_${_dateKey(d)}";
        final keyRef = _db.collection("booking_keys").doc(keyId);
        keyRefs.add(keyRef);
      }

      for (final keyRef in keyRefs) {
        final keySnap = await tx.get(keyRef);
        if (keySnap.exists) {
          final data = keySnap.data();
          final date = (data?["date"] ?? keyRef.id.split("_").last).toString();
          throw Exception("This labour is already booked on $date");
        }
      }

      // 2) lock each day (one booking per labour per day)
      for (int i = 0; i < days.length; i++) {
        final d = days[i];
        final keyRef = keyRefs[i];
        tx.set(keyRef, {
          "labourId": labourId,
          "date": _dateKey(d),
          "farmerUid": farmerUid, // needed for delete rule
          "createdAt": FieldValue.serverTimestamp(),
        });
      }

      // 3) create booking doc
      final bookingRef = _db.collection("bookings").doc();
      tx.set(bookingRef, {
        "bookingId": bookingRef.id,
        "farmerUid": farmerUid,
        "farmerName": farmerName,

        // ✅ IMPORTANT: store farmer contact so labour can display without reading users doc
        "farmerPhone": farmerPhone,
        "farmerEmail": farmerEmail,

        "labourId": labourId,
        "labourUid": labourUid,
        "startDate": _dateKey(start),
        "endDate": _dateKey(end),
        "type": halfDay ? "half-day" : "day",
        "note": (note ?? "").trim(),
        "status": "pending", // pending / accepted / rejected / cancelled
        "createdAt": FieldValue.serverTimestamp(),
      });
    });
  }

  /// Farmer bookings
  static Stream<QuerySnapshot<Map<String, dynamic>>> myBookings() {
    final u = _auth.currentUser;
    if (u == null) return const Stream.empty();
    return _db
        .collection("bookings")
        .where("farmerUid", isEqualTo: u.uid)
        .snapshots();
  }

  /// Labour incoming bookings
  static Stream<QuerySnapshot<Map<String, dynamic>>> labourBookings({
    required String labourId,
  }) {
    return _db
        .collection("bookings")
        .where("labourId", isEqualTo: labourId)
        .snapshots();
  }

  /// Update status (Labour accept/reject OR Farmer cancel)
  static Future<void> updateStatus({
    required String bookingId,
    required String status, // accepted / rejected / cancelled
  }) async {
    final bookingRef = _db.collection("bookings").doc(bookingId);

    await _db.runTransaction((tx) async {
      final snap = await tx.get(bookingRef);
      if (!snap.exists) throw Exception("Booking not found");

      final data = snap.data() as Map<String, dynamic>;
      final labourId = (data["labourId"] ?? "").toString();
      final startStr = (data["startDate"] ?? "").toString();
      final endStr = (data["endDate"] ?? "").toString();

      // update booking status
      tx.update(bookingRef, {
        "status": status,
        "updatedAt": FieldValue.serverTimestamp(),
      });

      // if rejected/cancelled => remove locks so other bookings can happen
      if (status == "rejected" || status == "cancelled") {
        DateTime parse(String s) {
          final p = s.split("-");
          return DateTime(int.parse(p[0]), int.parse(p[1]), int.parse(p[2]));
        }

        final start = parse(startStr);
        final end = parse(endStr);
        final days = _daysInRange(start, end);

        for (final d in days) {
          final keyId = "${labourId}_${_dateKey(d)}";
          final keyRef = _db.collection("booking_keys").doc(keyId);
          tx.delete(keyRef);
        }
      }
    });
  }
}
