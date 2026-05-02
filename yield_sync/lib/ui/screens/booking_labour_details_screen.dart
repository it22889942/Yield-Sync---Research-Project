import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import '../../utils/app_colors.dart';

class BookingLabourDetailsScreen extends StatelessWidget {
  final String labourId;

  final String? startDate; // "2026-02-16"
  final String? endDate; // "2026-02-16"
  final String? type; // "day" / "half-day"

  final String? labourUid;

  const BookingLabourDetailsScreen({
    super.key,
    required this.labourId,
    this.startDate,
    this.endDate,
    this.type,
    this.labourUid,
  });

  Future<_LabourBundle> _loadAll() async {
    final db = FirebaseFirestore.instance;

    final labourSnap = await db.collection("labours").doc(labourId).get();

    DocumentSnapshot<Map<String, dynamic>>? userSnap;

    if (labourUid != null && labourUid!.trim().isNotEmpty) {
      userSnap = await db.collection("users").doc(labourUid).get();
    } else {
      final q = await db
          .collection("users")
          .where("userType", isEqualTo: "labour")
          .where("labourId", isEqualTo: labourId)
          .limit(1)
          .get();

      if (q.docs.isNotEmpty) userSnap = q.docs.first;
    }

    return _LabourBundle(labourSnap: labourSnap, userSnap: userSnap);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.surface,
      body: SafeArea(
        child: Column(
          children: [
            _HeroHeader(
              title: "Labour Details",
              subtitle: "Booking information & contact",
              onBack: () => Navigator.pop(context),
            ),
            Expanded(
              child: FutureBuilder<_LabourBundle>(
                future: _loadAll(),
                builder: (context, snap) {
                  if (snap.connectionState == ConnectionState.waiting) {
                    return const Center(child: CircularProgressIndicator());
                  }

                  if (snap.hasError) {
                    return _StateBox(
                      icon: Icons.error_outline_rounded,
                      title: "Something went wrong",
                      subtitle: "${snap.error}",
                      isError: true,
                    );
                  }

                  final bundle = snap.data;
                  if (bundle == null) {
                    return const _StateBox(
                      icon: Icons.info_outline_rounded,
                      title: "No data",
                      subtitle: "Unable to load labour details.",
                    );
                  }

                  final labourDoc = bundle.labourSnap;
                  if (!labourDoc.exists) {
                    return _StateBox(
                      icon: Icons.search_off_rounded,
                      title: "Not found",
                      subtitle: "Labour not found: $labourId",
                    );
                  }

                  final labour = labourDoc.data() ?? {};
                  final user = bundle.userSnap?.data() ?? {};

                  final id =
                      (labour["Labour_ID"] ?? labour["labourId"] ?? labourId)
                          .toString();
                  final name =
                      (labour["Name"] ?? labour["name"] ?? "—").toString();
                  final labourType =
                      (labour["Labour_Type"] ?? labour["labourType"] ?? "—")
                          .toString();
                  final rate =
                      (labour["Hourly_Rate"] ?? labour["hourlyRate"] ?? "—")
                          .toString();
                  final location =
                      (labour["Location"] ?? labour["location"] ?? "—")
                          .toString();

                  final phone = (user["phone"] ?? "—").toString();
                  final email = (user["email"] ?? "—").toString();

                  final bStart = (startDate ?? "").trim();
                  final bEnd = (endDate ?? "").trim();
                  final bType = (type ?? "").trim();

                  return Stack(
                    children: [
                      ListView(
                        padding: const EdgeInsets.fromLTRB(16, 12, 16, 100),
                        children: [
                          // Booking summary (if passed)
                          if (bStart.isNotEmpty ||
                              bEnd.isNotEmpty ||
                              bType.isNotEmpty) ...[
                            _SectionTitle(
                              icon: Icons.receipt_long_rounded,
                              title: "Booking Summary",
                            ),
                            const SizedBox(height: 10),
                            _Card(
                              child: Column(
                                children: [
                                  _InfoRow(
                                    icon: Icons.date_range_rounded,
                                    label: "Dates",
                                    value: _fmtDates(bStart, bEnd),
                                  ),
                                  const SizedBox(height: 10),
                                  _InfoRow(
                                    icon: Icons.category_rounded,
                                    label: "Type",
                                    value: bType.isEmpty ? "—" : bType,
                                  ),
                                ],
                              ),
                            ),
                            const SizedBox(height: 14),
                          ],

                          // Profile card
                          _SectionTitle(
                            icon: Icons.person_rounded,
                            title: "Worker Profile",
                          ),
                          const SizedBox(height: 10),
                          _Card(
                            child: Row(
                              children: [
                                CircleAvatar(
                                  radius: 28,
                                  backgroundColor:
                                      AppColors.primary.withOpacity(0.20),
                                  child: Text(
                                    (name.isNotEmpty && name != "—")
                                        ? name[0].toUpperCase()
                                        : "W",
                                    style: const TextStyle(
                                      fontWeight: FontWeight.w900,
                                      color: AppColors.textDark,
                                      fontSize: 20,
                                    ),
                                  ),
                                ),
                                const SizedBox(width: 12),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        name,
                                        maxLines: 1,
                                        overflow: TextOverflow.ellipsis,
                                        style: const TextStyle(
                                          fontWeight: FontWeight.w900,
                                          color: AppColors.textDark,
                                          fontSize: 16,
                                        ),
                                      ),
                                      const SizedBox(height: 6),
                                      Wrap(
                                        spacing: 8,
                                        runSpacing: 8,
                                        children: [
                                          _Pill(
                                            icon: Icons.badge_rounded,
                                            text: labourType,
                                          ),
                                          _Pill(
                                            icon: Icons.location_on_rounded,
                                            text: location,
                                          ),
                                        ],
                                      )
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          ),

                          const SizedBox(height: 12),

                          // Details card
                          _SectionTitle(
                            icon: Icons.info_outline_rounded,
                            title: "Details",
                          ),
                          const SizedBox(height: 10),
                          _Card(
                            child: Column(
                              children: [
                                _InfoRow(
                                  icon: Icons.fingerprint_rounded,
                                  label: "Labour ID",
                                  value: id,
                                ),
                                const SizedBox(height: 10),
                                _InfoRow(
                                  icon: Icons.payments_rounded,
                                  label: "Hourly Rate",
                                  value: rate,
                                ),
                                const SizedBox(height: 10),
                                _InfoRow(
                                  icon: Icons.map_rounded,
                                  label: "Location",
                                  value: location,
                                ),
                              ],
                            ),
                          ),

                          const SizedBox(height: 12),

                          // Contact + quick actions
                          _SectionTitle(
                            icon: Icons.call_rounded,
                            title: "Contact",
                          ),
                          const SizedBox(height: 10),
                          _Card(
                            child: Column(
                              children: [
                                _InfoRow(
                                  icon: Icons.phone_rounded,
                                  label: "Phone",
                                  value: phone,
                                ),
                                const SizedBox(height: 10),
                                _InfoRow(
                                  icon: Icons.email_rounded,
                                  label: "Email",
                                  value: email,
                                ),
                                const SizedBox(height: 12),
                                Row(
                                  children: [
                                    Expanded(
                                      child: _ActionBtn(
                                        icon: Icons.call_rounded,
                                        label: "Call",
                                        onTap: phone == "—"
                                            ? null
                                            : () {
                                                // UI only: add url_launcher later
                                              },
                                      ),
                                    ),
                                    const SizedBox(width: 10),
                                    Expanded(
                                      child: _ActionBtn(
                                        icon: Icons.chat_rounded,
                                        label: "WhatsApp",
                                        onTap: phone == "—"
                                            ? null
                                            : () {
                                                // UI only: add url_launcher later
                                              },
                                      ),
                                    ),
                                    const SizedBox(width: 10),
                                    Expanded(
                                      child: _ActionBtn(
                                        icon: Icons.mail_rounded,
                                        label: "Email",
                                        onTap: email == "—"
                                            ? null
                                            : () {
                                                // UI only: add url_launcher later
                                              },
                                      ),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),

                      // Sticky Back button (premium UX)
                      Positioned(
                        left: 16,
                        right: 16,
                        bottom: 12,
                        child: SafeArea(
                          top: false,
                          child: Container(
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(18),
                              border: Border.all(color: AppColors.border),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withOpacity(0.10),
                                  blurRadius: 22,
                                  offset: const Offset(0, 14),
                                ),
                              ],
                            ),
                            child: SizedBox(
                              height: 48,
                              child: ElevatedButton.icon(
                                onPressed: () => Navigator.pop(context),
                                icon: const Icon(Icons.arrow_back_rounded),
                                label: const Text(
                                  "Back",
                                  style: TextStyle(fontWeight: FontWeight.w900),
                                ),
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: AppColors.primary,
                                  foregroundColor: AppColors.darkGreen,
                                  elevation: 0,
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(14),
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                    ],
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  static String _fmtDates(String start, String end) {
    if (start.isEmpty && end.isEmpty) return "—";
    if (start.isNotEmpty && end.isEmpty) return start;
    if (start.isEmpty && end.isNotEmpty) return end;
    return "$start → $end";
  }
}

// ======================= UI PARTS =======================

class _HeroHeader extends StatelessWidget {
  final String title;
  final String subtitle;
  final VoidCallback onBack;

  const _HeroHeader({
    required this.title,
    required this.subtitle,
    required this.onBack,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 18),
      decoration: const BoxDecoration(
        gradient: AppColors.heroGradient,
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(28),
          bottomRight: Radius.circular(28),
        ),
      ),
      child: Column(
        children: [
          Row(
            children: [
              InkWell(
                borderRadius: BorderRadius.circular(999),
                onTap: onBack,
                child: CircleAvatar(
                  radius: 18,
                  backgroundColor: Colors.white.withOpacity(0.14),
                  child: const Icon(Icons.arrow_back_ios_new_rounded,
                      color: Colors.white),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  title,
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.95),
                    fontWeight: FontWeight.w900,
                    fontSize: 16,
                  ),
                ),
              ),
              CircleAvatar(
                radius: 18,
                backgroundColor: Colors.white.withOpacity(0.14),
                child: const Icon(Icons.person_rounded, color: Colors.white),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Align(
            alignment: Alignment.centerLeft,
            child: Text(
              subtitle,
              style: TextStyle(
                color: Colors.white.withOpacity(0.78),
                fontWeight: FontWeight.w700,
                fontSize: 12.8,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _SectionTitle extends StatelessWidget {
  final IconData icon;
  final String title;
  const _SectionTitle({required this.icon, required this.title});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          width: 34,
          height: 34,
          decoration: BoxDecoration(
            color: AppColors.primary.withOpacity(0.14),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: AppColors.border),
          ),
          child: Icon(icon, size: 18, color: AppColors.darkGreen),
        ),
        const SizedBox(width: 10),
        Text(
          title,
          style: const TextStyle(
            fontWeight: FontWeight.w900,
            color: AppColors.textDark,
            fontSize: 15,
          ),
        ),
      ],
    );
  }
}

class _Card extends StatelessWidget {
  final Widget child;
  const _Card({required this.child});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
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
}

class _InfoRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;

  const _InfoRow({
    required this.icon,
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          width: 36,
          height: 36,
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AppColors.border),
          ),
          child: Icon(icon, size: 18, color: AppColors.darkGreen),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Text(
            label,
            style: TextStyle(
              fontWeight: FontWeight.w800,
              color: AppColors.textDark.withOpacity(0.65),
            ),
          ),
        ),
        const SizedBox(width: 10),
        Flexible(
          child: Text(
            value,
            textAlign: TextAlign.right,
            style: const TextStyle(
              fontWeight: FontWeight.w900,
              color: AppColors.textDark,
            ),
          ),
        ),
      ],
    );
  }
}

class _Pill extends StatelessWidget {
  final IconData icon;
  final String text;

  const _Pill({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: AppColors.darkGreen),
          const SizedBox(width: 6),
          ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 180),
            child: Text(
              text.isEmpty ? "—" : text,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                fontWeight: FontWeight.w800,
                color: AppColors.textDark.withOpacity(0.75),
                fontSize: 12,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ActionBtn extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback? onTap;

  const _ActionBtn({
    required this.icon,
    required this.label,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final disabled = onTap == null;

    return InkWell(
      borderRadius: BorderRadius.circular(16),
      onTap: onTap,
      child: Container(
        height: 46,
        decoration: BoxDecoration(
          color: disabled
              ? AppColors.surface
              : AppColors.primary.withOpacity(0.14),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppColors.border),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              size: 18,
              color: disabled
                  ? AppColors.textDark.withOpacity(0.35)
                  : AppColors.darkGreen,
            ),
            const SizedBox(width: 8),
            Text(
              label,
              style: TextStyle(
                fontWeight: FontWeight.w900,
                color: disabled
                    ? AppColors.textDark.withOpacity(0.35)
                    : AppColors.textDark,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _StateBox extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final bool isError;

  const _StateBox({
    super.key,
    required this.icon,
    required this.title,
    required this.subtitle,
    this.isError = false,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Container(
        margin: const EdgeInsets.all(16),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: isError ? Colors.red.withOpacity(0.06) : Colors.white,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(
            color: isError ? Colors.red.withOpacity(0.20) : AppColors.border,
          ),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon,
                size: 34, color: isError ? Colors.red : AppColors.darkGreen),
            const SizedBox(height: 10),
            Text(
              title,
              style: const TextStyle(
                fontWeight: FontWeight.w900,
                color: AppColors.textDark,
                fontSize: 16,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              subtitle,
              textAlign: TextAlign.center,
              style: TextStyle(
                fontWeight: FontWeight.w700,
                color: AppColors.textDark.withOpacity(0.65),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _LabourBundle {
  final DocumentSnapshot<Map<String, dynamic>> labourSnap;
  final DocumentSnapshot<Map<String, dynamic>>? userSnap;

  _LabourBundle({required this.labourSnap, required this.userSnap});
}
