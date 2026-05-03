import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

/// Matches Firestore `moderation_status` (see admin approval flow).
enum PostModerationState {
  pending,
  approved,
  rejected,
}

/// Service for logged-in users to add a labour post or equipment post.
/// New docs appear in lists and in recommendations after next model retrain.
class LabourEquipmentPostService {
  static final FirebaseFirestore _db = FirebaseFirestore.instance;
  static final FirebaseAuth _auth = FirebaseAuth.instance;

  /// Parse moderation from a Firestore document map. Missing field = legacy = approved.
  static PostModerationState moderationStateFromData(Map<String, dynamic>? data) {
    final raw = (data?['moderation_status'] ?? data?['Moderation_Status'] ?? '')
        .toString()
        .trim()
        .toLowerCase();
    switch (raw) {
      case 'pending':
        return PostModerationState.pending;
      case 'rejected':
        return PostModerationState.rejected;
      case 'approved':
      case '':
        return PostModerationState.approved;
      default:
        return PostModerationState.approved;
    }
  }

  /// Create a new labour post. Returns the new document ID (Labour_ID).
  static Future<String> createLabourPost({
    required String name,
    required String labourType,
    required String skillLevel,
    required String location,
    required String season,
    required String cropType,
    required String availableDay,
    required String availableTime,
    required double hourlyRate,
    int experienceYears = 0,
    int jobsCompleted = 0,
    double rating = 5.0,
  }) async {
    final user = _auth.currentUser;
    if (user == null) throw Exception('You must be logged in to add a labour post');

    final profile = await _db.collection('users').doc(user.uid).get();
    final profileData = profile.data() ?? {};
    final ownerName = name.trim().isNotEmpty
        ? name
        : (profileData['firstName'] ?? '') + ' ' + (profileData['lastName'] ?? '').trim();
    final phone = (profileData['phone'] ?? '').toString();
    final email = (profileData['email'] ?? user.email ?? '').toString();

    final ref = _db.collection('labours').doc();
    final labourId = ref.id;

    await ref.set({
      'Labour_ID': labourId,
      'labourId': labourId,
      'Name': name.trim(),
      'Labour_Type': labourType.trim(),
      'Skill_Level': skillLevel.trim(),
      'Location': location.trim(),
      'Season': season.trim(),
      'Crop_Type': cropType.trim(),
      'Available_Day': availableDay.trim(),
      'Available_Time': availableTime.trim(),
      'Hourly_Rate': hourlyRate,
      'Rating': rating,
      'Experience_Years': experienceYears,
      'Jobs_Completed': jobsCompleted,
      'createdByUid': user.uid,
      'ownerPhone': phone,
      'ownerEmail': email,
      'moderation_status': 'pending',
      'createdAt': FieldValue.serverTimestamp(),
    });

    return labourId;
  }

  /// Create a new equipment post. Returns the new document ID (Equipment_ID).
  static Future<String> createEquipmentPost({
    required String equipmentType,
    required String forCrop,
    required String nearestMajorDistrict,
    required String season,
    required String condition,
    required String availableDay,
    required String availableTime,
    required double hourlyRateLkr,
    required double dailyRateLkr,
    String ownerName = '',
    String ownerContact = '',
    String ownerType = 'individual',
    int pastBookings = 0,
    double successRatePct = 100.0,
    double rating = 5.0,
  }) async {
    final user = _auth.currentUser;
    if (user == null) throw Exception('You must be logged in to list equipment');

    final profile = await _db.collection('users').doc(user.uid).get();
    final profileData = profile.data() ?? {};
    final name = ownerName.trim().isNotEmpty
        ? ownerName
        : '${profileData['firstName'] ?? ''} ${profileData['lastName'] ?? ''}'.trim();
    final contact = ownerContact.trim().isNotEmpty
        ? ownerContact
        : (profileData['phone'] ?? '').toString();
    final finalOwnerName = name.isEmpty ? 'Owner' : name;

    final ref = _db.collection('equipments').doc();
    final equipmentId = ref.id;

    await ref.set({
      'Equipment_ID': equipmentId,
      'equipmentId': equipmentId,
      'Equipment_Type': equipmentType.trim(),
      'For_Crop': forCrop.trim(),
      'Nearest_Major_District': nearestMajorDistrict.trim(),
      'Season': season.trim(),
      'Condition': condition.trim(),
      'Available_Day': availableDay.trim(),
      'Available_Time': availableTime.trim(),
      'Equipment_Owner_Name': finalOwnerName,
      'Owner_Contact_No': contact,
      'Owner_Type': ownerType.trim(),
      'Hourly_Rate_LKR': hourlyRateLkr,
      'Daily_Rate_LKR': dailyRateLkr,
      'Rating': rating,
      'Past_Bookings': pastBookings,
      'Success_Rate_pct': successRatePct,
      'createdByUid': user.uid,
      'moderation_status': 'pending',
      'createdAt': FieldValue.serverTimestamp(),
    });

    return equipmentId;
  }

  /// Stream of labour posts created by the current user.
  static Stream<QuerySnapshot<Map<String, dynamic>>> myLabourPostsStream() {
    final user = _auth.currentUser;
    if (user == null) return const Stream.empty();
    return _db
        .collection('labours')
        .where('createdByUid', isEqualTo: user.uid)
        .snapshots();
  }

  /// Stream of equipment posts created by the current user.
  static Stream<QuerySnapshot<Map<String, dynamic>>> myEquipmentPostsStream() {
    final user = _auth.currentUser;
    if (user == null) return const Stream.empty();
    return _db
        .collection('equipments')
        .where('createdByUid', isEqualTo: user.uid)
        .snapshots();
  }
}
