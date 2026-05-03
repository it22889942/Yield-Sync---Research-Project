import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

import '../../utils/app_colors.dart';
import '../../services/auth_service.dart';
import '../../services/app_routes.dart';
import '../../services/nav.dart';
import '../../services/review_service.dart';
import '../widgets/app_text_field.dart';
import '../widgets/primary_button.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final _formKey = GlobalKey<FormState>();

  final _firstCtrl = TextEditingController();
  final _lastCtrl = TextEditingController();
  final _userCtrl = TextEditingController();
  final _phoneCtrl = TextEditingController();

  final _firstFocus = FocusNode();
  final _lastFocus = FocusNode();
  final _userFocus = FocusNode();
  final _phoneFocus = FocusNode();

  String _email = "";
  String _typeText = "Farmer";
  bool _loading = true;
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    _loadProfile();
  }

  @override
  void dispose() {
    _firstCtrl.dispose();
    _lastCtrl.dispose();
    _userCtrl.dispose();
    _phoneCtrl.dispose();

    _firstFocus.dispose();
    _lastFocus.dispose();
    _userFocus.dispose();
    _phoneFocus.dispose();

    super.dispose();
  }

  void _unfocus() {
    if (!mounted) return;
    final scope = FocusScope.of(context);
    if (!scope.hasPrimaryFocus) scope.unfocus();
  }

  Future<void> _loadProfile() async {
    setState(() => _loading = true);

    final data = await AuthService().getMyProfile();
    if (!mounted) return;

    if (data == null) {
      setState(() => _loading = false);
      return;
    }

    _firstCtrl.text = (data["firstName"] ?? "").toString();
    _lastCtrl.text = (data["lastName"] ?? "").toString();
    _userCtrl.text = (data["username"] ?? "").toString();
    _phoneCtrl.text = (data["phone"] ?? "").toString();
    _email = (data["email"] ?? "").toString();

    final t = (data["userType"] ?? "farmer").toString().toLowerCase();
    _typeText = (t == "labour")
        ? "Labour"
        : (t == "seller")
            ? "Seller"
            : (t == "admin")
                ? "Admin"
                : "Farmer";

    setState(() => _loading = false);
  }

  String? _req(String? v, String msg) {
    if (v == null || v.trim().isEmpty) return msg;
    return null;
  }

  Future<void> _save() async {
    _unfocus();
    if (!_formKey.currentState!.validate()) return;

    setState(() => _saving = true);

    try {
      await AuthService().updateMyProfile(
        firstName: _firstCtrl.text,
        lastName: _lastCtrl.text,
        username: _userCtrl.text,
        phone: _phoneCtrl.text,
      );

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Profile updated ✅")),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Update failed: $e")),
      );
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  Color _statusColor(String s) {
    switch (s) {
      case "accepted":
        return Colors.green;
      case "rejected":
        return Colors.red;
      case "cancelled":
        return Colors.orange;
      default:
        return Colors.blueGrey;
    }
  }

  Future<Map<String, dynamic>?> _getUserByUid(String uid) async {
    if (uid.trim().isEmpty) return null;
    final doc =
        await FirebaseFirestore.instance.collection("users").doc(uid).get();
    if (!doc.exists) return null;
    return doc.data();
  }

  Future<void> _showReviewDialog({
    required String bookingId,
    required String itemType, // "labour" or "equipment"
    required String itemId,
    required String itemName,
    required String ownerName,
  }) async {
    double rating = 5.0;
    String sentiment = "good";
    final ctrl = TextEditingController();

    await showDialog(
      context: context,
      barrierDismissible: true,
      builder: (_) {
        return StatefulBuilder(
          builder: (context, setStateDialog) {
            return Dialog(
              insetPadding: const EdgeInsets.all(14),
              backgroundColor: const Color(0xFFF2EEF6),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(22),
              ),
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 14),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(10),
                          decoration: BoxDecoration(
                            color: AppColors.primary.withOpacity(0.16),
                            borderRadius: BorderRadius.circular(16),
                            border:
                                Border.all(color: AppColors.primary, width: 1),
                          ),
                          child: const Icon(Icons.rate_review_rounded,
                              color: AppColors.darkGreen),
                        ),
                        const SizedBox(width: 12),
                        const Expanded(
                          child: Text(
                            "Add Review",
                            style: TextStyle(
                              fontWeight: FontWeight.w900,
                              fontSize: 18,
                              color: AppColors.textDark,
                            ),
                          ),
                        ),
                        IconButton(
                          onPressed: () => Navigator.pop(context),
                          icon: const Icon(Icons.close_rounded),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(18),
                        border: Border.all(color: AppColors.border),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.04),
                            blurRadius: 14,
                            offset: const Offset(0, 10),
                          ),
                        ],
                      ),
                      child: Column(
                        children: [
                          _kv("Booking ID", bookingId),
                          _kv("Type", itemType.toUpperCase()),
                          _kv("ID", itemId),
                          _kv("Item", itemName.isEmpty ? "—" : itemName),
                          _kv("Owner", ownerName.isEmpty ? "—" : ownerName),
                        ],
                      ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Expanded(
                          child: OutlinedButton(
                            onPressed: () =>
                                setStateDialog(() => sentiment = "good"),
                            style: OutlinedButton.styleFrom(
                              side: BorderSide(
                                color: sentiment == "good"
                                    ? Colors.green
                                    : AppColors.border,
                              ),
                              backgroundColor: sentiment == "good"
                                  ? Colors.green.withOpacity(0.08)
                                  : Colors.white,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(16),
                              ),
                            ),
                            child: Text(
                              "Good ✅",
                              style: TextStyle(
                                fontWeight: FontWeight.w900,
                                color: sentiment == "good"
                                    ? Colors.green
                                    : AppColors.textDark,
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: OutlinedButton(
                            onPressed: () =>
                                setStateDialog(() => sentiment = "bad"),
                            style: OutlinedButton.styleFrom(
                              side: BorderSide(
                                color: sentiment == "bad"
                                    ? Colors.red
                                    : AppColors.border,
                              ),
                              backgroundColor: sentiment == "bad"
                                  ? Colors.red.withOpacity(0.08)
                                  : Colors.white,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(16),
                              ),
                            ),
                            child: Text(
                              "Bad ❌",
                              style: TextStyle(
                                fontWeight: FontWeight.w900,
                                color: sentiment == "bad"
                                    ? Colors.red
                                    : AppColors.textDark,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Text(
                          "Rating",
                          style: TextStyle(
                            fontWeight: FontWeight.w900,
                            color: AppColors.textDark.withOpacity(0.85),
                          ),
                        ),
                        const Spacer(),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 10, vertical: 6),
                          decoration: BoxDecoration(
                            color: AppColors.primary.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(999),
                            border: Border.all(color: AppColors.border),
                          ),
                          child: Text(
                            rating.toStringAsFixed(1),
                            style: const TextStyle(
                              fontWeight: FontWeight.w900,
                              color: AppColors.textDark,
                            ),
                          ),
                        ),
                      ],
                    ),
                    Slider(
                      value: rating,
                      min: 1,
                      max: 5,
                      divisions: 8,
                      onChanged: (v) => setStateDialog(() => rating = v),
                    ),
                    const SizedBox(height: 6),
                    TextField(
                      controller: ctrl,
                      maxLines: 3,
                      decoration: InputDecoration(
                        hintText: "Write your review...",
                        filled: true,
                        fillColor: Colors.white,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(16),
                          borderSide: const BorderSide(color: AppColors.border),
                        ),
                        enabledBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(16),
                          borderSide: const BorderSide(color: AppColors.border),
                        ),
                      ),
                    ),
                    const SizedBox(height: 14),
                    Row(
                      children: [
                        Expanded(
                          child: TextButton(
                            onPressed: () => Navigator.pop(context),
                            child: const Text(
                              "Cancel",
                              style: TextStyle(fontWeight: FontWeight.w900),
                            ),
                          ),
                        ),
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: () async {
                              try {
                                final comment = ctrl.text.trim();
                                if (comment.isEmpty) {
                                  ScaffoldMessenger.of(this.context)
                                      .showSnackBar(
                                    const SnackBar(
                                      content: Text("Please write a review."),
                                    ),
                                  );
                                  return;
                                }

                                await ReviewService.addReview(
                                  bookingId: bookingId,
                                  itemType: itemType,
                                  itemId: itemId,
                                  itemName: itemName,
                                  ownerName: ownerName,
                                  sentiment: sentiment,
                                  rating: rating,
                                  comment: comment,
                                );

                                if (context.mounted) Navigator.pop(context);

                                if (mounted) {
                                  ScaffoldMessenger.of(this.context)
                                      .showSnackBar(
                                    const SnackBar(
                                      content: Text("Review saved ✅"),
                                    ),
                                  );
                                }
                              } catch (e) {
                                if (mounted) {
                                  ScaffoldMessenger.of(this.context)
                                      .showSnackBar(
                                    SnackBar(
                                      content: Text("Save failed: $e"),
                                    ),
                                  );
                                }
                              }
                            },
                            icon: const Icon(Icons.send_rounded),
                            label: const Text(
                              "Submit",
                              style: TextStyle(fontWeight: FontWeight.w900),
                            ),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: AppColors.primary,
                              foregroundColor: AppColors.darkGreen,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(999),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  Future<void> _showEquipmentAcceptedPopup({
    required Map<String, dynamic> booking,
    required String bookingId,
  }) async {
    final equipmentOwnerUid = (booking["equipmentOwnerUid"] ?? "").toString();
    final equipmentId = (booking["equipmentId"] ?? "").toString();
    final start = (booking["startDate"] ?? "").toString();
    final end = (booking["endDate"] ?? "").toString();
    final type = (booking["type"] ?? "").toString();
    final status = (booking["status"] ?? "").toString();

    showDialog(
      context: context,
      barrierDismissible: true,
      builder: (_) {
        return Dialog(
          insetPadding: const EdgeInsets.all(14),
          backgroundColor: const Color(0xFFF2EEF6),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(22)),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 14),
            child: FutureBuilder<Map<String, dynamic>?>(
              future: _getUserByUid(equipmentOwnerUid),
              builder: (context, snap) {
                final ownerNameFromBooking =
                    (booking["equipmentOwnerName"] ?? "").toString();
                final ownerPhoneFromBooking =
                    (booking["equipmentOwnerPhone"] ?? "").toString();
                final ownerEmailFromBooking =
                    (booking["equipmentOwnerEmail"] ?? "").toString();

                final u = snap.data ?? {};
                final ownerNameFromUser =
                    (u["username"] ?? u["firstName"] ?? u["name"] ?? "")
                        .toString();
                final ownerPhoneFromUser = (u["phone"] ?? "").toString();
                final ownerEmailFromUser = (u["email"] ?? "").toString();

                final ownerName = ownerNameFromBooking.isNotEmpty
                    ? ownerNameFromBooking
                    : (ownerNameFromUser.isNotEmpty
                        ? ownerNameFromUser
                        : "Owner");

                final ownerPhone = ownerPhoneFromBooking.isNotEmpty
                    ? ownerPhoneFromBooking
                    : (ownerPhoneFromUser.isNotEmpty
                        ? ownerPhoneFromUser
                        : "—");

                final ownerEmail = ownerEmailFromBooking.isNotEmpty
                    ? ownerEmailFromBooking
                    : (ownerEmailFromUser.isNotEmpty
                        ? ownerEmailFromUser
                        : "—");

                return Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(10),
                          decoration: BoxDecoration(
                            color: AppColors.primary.withOpacity(0.16),
                            borderRadius: BorderRadius.circular(16),
                            border:
                                Border.all(color: AppColors.primary, width: 1),
                          ),
                          child: const Icon(Icons.agriculture_rounded,
                              color: AppColors.darkGreen),
                        ),
                        const SizedBox(width: 12),
                        const Expanded(
                          child: Text(
                            "Equipment Accepted",
                            style: TextStyle(
                              fontWeight: FontWeight.w900,
                              fontSize: 18,
                              color: AppColors.textDark,
                            ),
                          ),
                        ),
                        IconButton(
                          onPressed: () => Navigator.pop(context),
                          icon: const Icon(Icons.close_rounded),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    _modernCard(
                      child: Column(
                        children: [
                          _kv("Booking ID", bookingId),
                          _kv("Status", status.toUpperCase()),
                          _kv("Equipment ID", equipmentId),
                          _kv("Dates", "$start → $end"),
                          _kv("Type", type.isEmpty ? "—" : type),
                        ],
                      ),
                    ),
                    const SizedBox(height: 12),
                    _modernCard(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            "Owner Details",
                            style: TextStyle(
                              fontWeight: FontWeight.w900,
                              color: AppColors.textDark,
                            ),
                          ),
                          const SizedBox(height: 10),
                          _kv("Name", ownerName),
                          _kv("Contact No", ownerPhone),
                          _kv("Email", ownerEmail),
                        ],
                      ),
                    ),
                    const SizedBox(height: 14),
                    Align(
                      alignment: Alignment.centerRight,
                      child: ElevatedButton.icon(
                        onPressed: () => Navigator.pop(context),
                        icon: const Icon(Icons.close_rounded),
                        label: const Text(
                          "Close",
                          style: TextStyle(fontWeight: FontWeight.w900),
                        ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppColors.primary,
                          foregroundColor: AppColors.darkGreen,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(999),
                          ),
                        ),
                      ),
                    ),
                  ],
                );
              },
            ),
          ),
        );
      },
    );
  }

  // ===== Modern UI helpers =====

  static Widget _modernCard({required Widget child}) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 18,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: child,
    );
  }

  static Widget _kv(String k, String v) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Expanded(
            child: Text(
              k,
              style: TextStyle(
                fontWeight: FontWeight.w800,
                color: AppColors.textDark.withOpacity(0.70),
              ),
            ),
          ),
          Expanded(
            child: Text(
              v,
              textAlign: TextAlign.right,
              style: const TextStyle(
                fontWeight: FontWeight.w900,
                color: AppColors.textDark,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _actionCard({
    required IconData icon,
    required String label,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(22),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(22),
        child: Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(22),
            border: Border.all(color: AppColors.border),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.05),
                blurRadius: 18,
                offset: const Offset(0, 12),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.14),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: AppColors.border),
                ),
                child: Icon(icon, size: 22, color: AppColors.darkGreen),
              ),
              const SizedBox(height: 10),
              Text(
                label,
                style: const TextStyle(
                  fontWeight: FontWeight.w900,
                  fontSize: 13,
                  color: AppColors.textDark,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                subtitle,
                style: TextStyle(
                  fontWeight: FontWeight.w700,
                  fontSize: 11,
                  color: AppColors.textDark.withOpacity(0.6),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _sectionTitle(String title, {IconData? icon}) {
    return Row(
      children: [
        if (icon != null) ...[
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: AppColors.primary.withOpacity(0.14),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: AppColors.border),
            ),
            child: Icon(icon, size: 18, color: AppColors.darkGreen),
          ),
          const SizedBox(width: 10),
        ],
        Expanded(
          child: Text(
            title,
            style: const TextStyle(
              fontWeight: FontWeight.w900,
              fontSize: 16,
              color: AppColors.textDark,
            ),
          ),
        ),
      ],
    );
  }

  Widget _header() {
    final initials = ((_firstCtrl.text.trim().isNotEmpty
                ? _firstCtrl.text.trim()[0]
                : "") +
            (_lastCtrl.text.trim().isNotEmpty ? _lastCtrl.text.trim()[0] : ""))
        .toUpperCase();

    return Container(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
      decoration: BoxDecoration(
        gradient: AppColors.heroGradient,
        borderRadius: BorderRadius.circular(26),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.10),
            blurRadius: 24,
            offset: const Offset(0, 16),
          ),
        ],
      ),
      child: Row(
        children: [
          InkWell(
            borderRadius: BorderRadius.circular(999),
            onTap: () => Nav.back(context),
            child: Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.14),
                borderRadius: BorderRadius.circular(999),
                border: Border.all(color: Colors.white.withOpacity(0.18)),
              ),
              child: const Icon(Icons.arrow_back_ios_new_rounded,
                  color: Colors.white, size: 18),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  "My Profile",
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w900,
                    letterSpacing: -0.2,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  _email.isEmpty ? "—" : _email,
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.86),
                    fontWeight: FontWeight.w700,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    _pill(
                      icon: Icons.verified_user_rounded,
                      text: _typeText,
                    ),
                    const SizedBox(width: 8),
                    _pill(
                      icon: Icons.phone_rounded,
                      text: _phoneCtrl.text.trim().isEmpty
                          ? "No phone"
                          : _phoneCtrl.text.trim(),
                    ),
                  ],
                )
              ],
            ),
          ),
          const SizedBox(width: 10),
          Container(
            width: 46,
            height: 46,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.18),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.white.withOpacity(0.20)),
            ),
            child: Center(
              child: Text(
                initials.isEmpty ? "U" : initials,
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w900,
                  fontSize: 16,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _pill({required IconData icon, required String text}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.14),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withOpacity(0.18)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: Colors.white),
          const SizedBox(width: 6),
          ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 170),
            child: Text(
              text.isEmpty ? "—" : text,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w800,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }

  Widget _simpleErr(String msg) {
    return Container(
      padding: const EdgeInsets.all(14),
      margin: const EdgeInsets.only(bottom: 10),
      decoration: BoxDecoration(
        color: Colors.red.withOpacity(0.06),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.red.withOpacity(0.2)),
      ),
      child: Text(
        msg,
        style: const TextStyle(color: Colors.red, fontWeight: FontWeight.w800),
      ),
    );
  }

  Widget _simpleInfo(String msg) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      child: Text(
        msg,
        style: TextStyle(
          color: AppColors.textDark.withOpacity(0.6),
          fontWeight: FontWeight.w800,
        ),
      ),
    );
  }

  Widget _bookingCard({
    required String title,
    required String status,
    required String start,
    required String end,
    required String type,
    VoidCallback? onCancel,
    required String acceptedHint,
  }) {
    final c = _statusColor(status);

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.045),
            blurRadius: 18,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    fontWeight: FontWeight.w900,
                    color: AppColors.textDark,
                  ),
                ),
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: c.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(999),
                  border: Border.all(color: c.withOpacity(0.25)),
                ),
                child: Text(
                  status.toUpperCase(),
                  style: TextStyle(
                    color: c,
                    fontWeight: FontWeight.w900,
                    fontSize: 11.5,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Icon(Icons.date_range_rounded,
                  size: 16, color: AppColors.textDark.withOpacity(0.70)),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  "$start → $end",
                  style: TextStyle(
                    fontWeight: FontWeight.w800,
                    color: AppColors.textDark.withOpacity(0.78),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Row(
            children: [
              Icon(Icons.category_rounded,
                  size: 16, color: AppColors.textDark.withOpacity(0.70)),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  type.isEmpty ? "—" : type,
                  style: TextStyle(
                    fontWeight: FontWeight.w800,
                    color: AppColors.textDark.withOpacity(0.78),
                  ),
                ),
              ),
            ],
          ),
          if (status == "accepted") ...[
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: AppColors.primary.withOpacity(0.10),
                borderRadius: BorderRadius.circular(18),
                border: Border.all(color: AppColors.border),
              ),
              child: Row(
                children: [
                  const Icon(Icons.touch_app_rounded,
                      size: 18, color: AppColors.darkGreen),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      acceptedHint,
                      style: TextStyle(
                        fontWeight: FontWeight.w800,
                        color: AppColors.textDark.withOpacity(0.70),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
          if (status == "pending" && onCancel != null) ...[
            const SizedBox(height: 10),
            Align(
              alignment: Alignment.centerRight,
              child: TextButton.icon(
                onPressed: onCancel,
                icon: const Icon(Icons.cancel_rounded, color: Colors.orange),
                label: const Text(
                  "Cancel",
                  style: TextStyle(
                    fontWeight: FontWeight.w900,
                    color: Colors.orange,
                  ),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      behavior: HitTestBehavior.opaque,
      onTap: _unfocus,
      child: Scaffold(
        backgroundColor: AppColors.surface,
        body: SafeArea(
          child: _loading
              ? const Center(child: CircularProgressIndicator())
              : RefreshIndicator(
                  onRefresh: _loadProfile,
                  child: NotificationListener<ScrollNotification>(
                    onNotification: (n) {
                      if (n is ScrollStartNotification) _unfocus();
                      return false;
                    },
                    child: ListView(
                      keyboardDismissBehavior:
                          ScrollViewKeyboardDismissBehavior.onDrag,
                      padding: const EdgeInsets.fromLTRB(18, 18, 18, 18),
                      children: [
                        _header(),
                        const SizedBox(height: 16),

                        // ===== Account Details =====
                        _sectionTitle("Account Details",
                            icon: Icons.person_rounded),
                        const SizedBox(height: 10),

                        _modernCard(
                          child: Form(
                            key: _formKey,
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                _ReadOnlyRow(label: "Email", value: _email),
                                const SizedBox(height: 10),
                                _ReadOnlyRow(
                                    label: "User Type", value: _typeText),
                                const SizedBox(height: 14),
                                Row(
                                  children: [
                                    Expanded(
                                      child: AppTextField(
                                        focusNode: _firstFocus,
                                        controller: _firstCtrl,
                                        label: "First Name",
                                        hint: "First name",
                                        validator: (v) =>
                                            _req(v, "First name required"),
                                        suffix:
                                            const Icon(Icons.badge_outlined),
                                      ),
                                    ),
                                    const SizedBox(width: 12),
                                    Expanded(
                                      child: AppTextField(
                                        focusNode: _lastFocus,
                                        controller: _lastCtrl,
                                        label: "Last Name",
                                        hint: "Last name",
                                        validator: (v) =>
                                            _req(v, "Last name required"),
                                        suffix:
                                            const Icon(Icons.badge_outlined),
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 12),
                                AppTextField(
                                  focusNode: _userFocus,
                                  controller: _userCtrl,
                                  label: "Username",
                                  hint: "username",
                                  validator: (v) {
                                    final r = _req(v, "Username required");
                                    if (r != null) return r;
                                    if (v!.trim().length < 3) {
                                      return "Minimum 3 characters";
                                    }
                                    return null;
                                  },
                                  suffix:
                                      const Icon(Icons.person_outline_rounded),
                                ),
                                const SizedBox(height: 12),
                                AppTextField(
                                  focusNode: _phoneFocus,
                                  controller: _phoneCtrl,
                                  label: "Contact Number",
                                  hint: "07XXXXXXXX",
                                  keyboardType: TextInputType.phone,
                                  validator: (v) => _req(v, "Phone required"),
                                  suffix: const Icon(Icons.phone_outlined),
                                ),
                                const SizedBox(height: 16),
                                PrimaryButton(
                                  text: _saving ? "Saving..." : "Save Changes",
                                  icon: _saving
                                      ? Icons.hourglass_top_rounded
                                      : Icons.save_rounded,
                                  onPressed: _saving ? null : _save,
                                ),
                                if (_saving) ...[
                                  const SizedBox(height: 10),
                                  Row(
                                    children: [
                                      const SizedBox(
                                        width: 18,
                                        height: 18,
                                        child: CircularProgressIndicator(
                                            strokeWidth: 2),
                                      ),
                                      const SizedBox(width: 10),
                                      Text(
                                        "Updating profile…",
                                        style: TextStyle(
                                          fontWeight: FontWeight.w800,
                                          color: AppColors.textDark
                                              .withOpacity(0.65),
                                        ),
                                      ),
                                    ],
                                  ),
                                ],
                              ],
                            ),
                          ),
                        ),

                        const SizedBox(height: 18),

                        // ================= Add labour / equipment posts =================
                        _sectionTitle("List your offer",
                            icon: Icons.add_circle_outline_rounded),
                        const SizedBox(height: 10),
                        Row(
                          children: [
                            Expanded(
                              child: _actionCard(
                                icon: Icons.person_add_rounded,
                                label: "Add as labour",
                                subtitle: "Offer your labour",
                                onTap: () => Navigator.pushNamed(
                                    context, AppRoutes.addLabourPost),
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: _actionCard(
                                icon: Icons.add_business_rounded,
                                label: "List equipment",
                                subtitle: "Rent out equipment",
                                onTap: () => Navigator.pushNamed(
                                    context, AppRoutes.addEquipmentPost),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 28),
                      ],
                    ),
                  ),
                ),
        ),
      ),
    );
  }
}

class _ReadOnlyRow extends StatelessWidget {
  final String label;
  final String value;

  const _ReadOnlyRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 12, 12, 12),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label,
              style: TextStyle(
                fontWeight: FontWeight.w800,
                color: AppColors.textDark.withOpacity(0.70),
              ),
            ),
          ),
          Expanded(
            child: Text(
              value.isEmpty ? "—" : value,
              textAlign: TextAlign.right,
              style: const TextStyle(
                fontWeight: FontWeight.w900,
                color: AppColors.textDark,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
