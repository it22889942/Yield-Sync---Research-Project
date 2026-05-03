import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import '../../services/review_service.dart';

import '../../services/booking_request_dialog.dart';
import '../../utils/app_colors.dart';
import '../widgets/app_shell.dart';
import 'labor_list_screen.dart';

class LaborDetailsScreen extends StatelessWidget {
  const LaborDetailsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final arg = ModalRoute.of(context)?.settings.arguments;
    final worker = (arg is LaborWorker) ? arg : null;

    return AppShell(
      currentIndex: 0,
      child: Scaffold(
        backgroundColor: AppColors.surface,
        body: SafeArea(
          child: worker == null
              ? _EmptyWorker(onBack: () => Navigator.pop(context))
              : Stack(
                  children: [
                    CustomScrollView(
                      slivers: [
                        SliverToBoxAdapter(child: _HeroHeader(worker: worker)),
                        SliverPadding(
                          padding: const EdgeInsets.fromLTRB(16, 12, 16, 120),
                          sliver: SliverList(
                            delegate: SliverChildListDelegate(
                              [
                                _IdentityCard(worker: worker),
                                const SizedBox(height: 12),

                                _Section(
                                  title: "Profile",
                                  icon: Icons.badge_rounded,
                                  child: Column(
                                    children: [
                                      _InfoTileModern(
                                        icon: Icons.badge_rounded,
                                        title: "Labour Type",
                                        value: worker.labourType.isEmpty
                                            ? "—"
                                            : worker.labourType,
                                      ),
                                      _InfoTileModern(
                                        icon: Icons.school_rounded,
                                        title: "Skill Level",
                                        value: worker.skillLevel.isEmpty
                                            ? "—"
                                            : worker.skillLevel,
                                      ),
                                      _InfoTileModern(
                                        icon: Icons.agriculture_rounded,
                                        title: "Crop Type",
                                        value: worker.cropType.isEmpty
                                            ? "—"
                                            : worker.cropType,
                                      ),
                                      _InfoTileModern(
                                        icon: Icons.wb_sunny_rounded,
                                        title: "Season",
                                        value: worker.season.isEmpty
                                            ? "—"
                                            : worker.season,
                                      ),
                                    ],
                                  ),
                                ),

                                const SizedBox(height: 12),

                                _Section(
                                  title: "Quick Stats",
                                  icon: Icons.auto_graph_rounded,
                                  child: Row(
                                    children: [
                                      Expanded(
                                        child: _StatPill(
                                          label: "Hourly Rate",
                                          value:
                                              "LKR ${worker.hourlyRate.toStringAsFixed(0)}",
                                          icon: Icons.payments_rounded,
                                        ),
                                      ),
                                      const SizedBox(width: 10),
                                      Expanded(
                                        child: _StatPill(
                                          label: "Jobs",
                                          value: "${worker.jobsCompleted}",
                                          icon: Icons.task_alt_rounded,
                                        ),
                                      ),
                                      const SizedBox(width: 10),
                                      Expanded(
                                        child: _StatPill(
                                          label: "Experience",
                                          value: "${worker.experienceYears}y",
                                          icon: Icons.timelapse_rounded,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),

                                const SizedBox(height: 12),

                                _Section(
                                  title: "Availability",
                                  icon: Icons.calendar_month_rounded,
                                  child: Row(
                                    children: [
                                      Expanded(
                                        child: _PillChip(
                                          icon: Icons.calendar_month_rounded,
                                          text: worker.availableDay.isEmpty
                                              ? "Day: —"
                                              : "Day: ${worker.availableDay}",
                                        ),
                                      ),
                                      const SizedBox(width: 10),
                                      Expanded(
                                        child: _PillChip(
                                          icon: Icons.schedule_rounded,
                                          text: worker.availableTime.isEmpty
                                              ? "Time: —"
                                              : "Time: ${worker.availableTime}",
                                        ),
                                      ),
                                    ],
                                  ),
                                ),

                                const SizedBox(height: 16),

                                // REVIEWS (Modern)
                                _LabourReviewsSectionModern(
                                  labourId: worker.id,
                                ),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),

                    // Sticky CTA
                    Positioned(
                      left: 16,
                      right: 16,
                      bottom: 12,
                      child: SafeArea(
                        top: false,
                        child: _StickyHireBar(
                          worker: worker,
                          onHire: () async {
                            final ok = await showDialog<bool>(
                              context: context,
                              builder: (_) =>
                                  BookingRequestDialog(labourId: worker.id),
                            );

                            if (ok == true && context.mounted) {
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                    content: Text("Request sent ✅ (Pending)")),
                              );
                            }
                          },
                        ),
                      ),
                    ),
                  ],
                ),
        ),
      ),
    );
  }
}

// ======================= HERO HEADER =======================
class _HeroHeader extends StatelessWidget {
  final LaborWorker worker;
  const _HeroHeader({required this.worker});

  String _initials(String name) {
    final n = name.trim();
    if (n.isEmpty) return "W";
    final parts = n.split(" ").where((e) => e.trim().isNotEmpty).toList();
    if (parts.length == 1) return parts.first[0].toUpperCase();
    return "${parts[0][0]}${parts[1][0]}".toUpperCase();
  }

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
              _roundIconBtn(
                icon: Icons.arrow_back_ios_new_rounded,
                onTap: () => Navigator.pop(context),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  "Worker Details",
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
              )
            ],
          ),
          const SizedBox(height: 14),

          // Top glass identity strip
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.12),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: Colors.white.withOpacity(0.16)),
            ),
            child: Row(
              children: [
                CircleAvatar(
                  radius: 26,
                  backgroundColor: Colors.white.withOpacity(0.18),
                  child: Text(
                    _initials(worker.name),
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w900,
                      fontSize: 18,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        worker.name.isEmpty ? "Worker" : worker.name,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w900,
                          fontSize: 16,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Row(
                        children: [
                          const Icon(Icons.location_on_rounded,
                              color: Colors.white70, size: 16),
                          const SizedBox(width: 4),
                          Expanded(
                            child: Text(
                              worker.location.isEmpty ? "—" : worker.location,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: TextStyle(
                                color: Colors.white.withOpacity(0.82),
                                fontWeight: FontWeight.w700,
                                fontSize: 12.2,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.14),
                    borderRadius: BorderRadius.circular(999),
                    border: Border.all(color: Colors.white.withOpacity(0.18)),
                  ),
                  child: Text(
                    "⭐ ${worker.rating.toStringAsFixed(1)}",
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w900,
                      fontSize: 12,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _roundIconBtn({required IconData icon, required VoidCallback onTap}) {
    return InkWell(
      borderRadius: BorderRadius.circular(999),
      onTap: onTap,
      child: CircleAvatar(
        radius: 18,
        backgroundColor: Colors.white.withOpacity(0.14),
        child: Icon(icon, color: Colors.white),
      ),
    );
  }
}

// ======================= MAIN IDENTITY CARD =======================
class _IdentityCard extends StatelessWidget {
  final LaborWorker worker;
  const _IdentityCard({required this.worker});

  @override
  Widget build(BuildContext context) {
    final chips = <String>[
      if (worker.skillLevel.isNotEmpty) "Skill: ${worker.skillLevel}",
      if (worker.cropType.isNotEmpty) "Crop: ${worker.cropType}",
      if (worker.labourType.isNotEmpty) worker.labourType,
    ];

    return _CardShell(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            "Overview",
            style: TextStyle(
              color: AppColors.textDark.withOpacity(0.65),
              fontWeight: FontWeight.w900,
            ),
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: chips.isEmpty
                ? [_TinyChip(text: "No tags")]
                : chips.map((e) => _TinyChip(text: e)).toList(),
          ),
          const SizedBox(height: 12),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: AppColors.primary.withOpacity(0.10),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppColors.border),
            ),
            child: Row(
              children: [
                const Icon(Icons.info_outline_rounded,
                    color: AppColors.darkGreen),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    "Tap “Hire / Request” to send a booking request. You’ll see status as Pending.",
                    style: TextStyle(
                      fontWeight: FontWeight.w700,
                      color: AppColors.textDark.withOpacity(0.70),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ======================= STICKY CTA =======================
class _StickyHireBar extends StatelessWidget {
  final LaborWorker worker;
  final VoidCallback onHire;
  const _StickyHireBar({required this.worker, required this.onHire});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.12),
            blurRadius: 22,
            offset: const Offset(0, 14),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  worker.name.isEmpty ? "Worker" : worker.name,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontWeight: FontWeight.w900,
                    color: AppColors.textDark,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  "LKR ${worker.hourlyRate.toStringAsFixed(0)}/hr • ⭐ ${worker.rating.toStringAsFixed(1)}",
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontWeight: FontWeight.w800,
                    color: AppColors.textDark.withOpacity(0.60),
                    fontSize: 12.2,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 10),
          SizedBox(
            height: 46,
            child: ElevatedButton.icon(
              onPressed: onHire,
              icon: const Icon(Icons.handshake_rounded),
              label: const Text(
                "Hire / Request",
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
        ],
      ),
    );
  }
}

// ======================= REVIEWS MODERN =======================
class _LabourReviewsSectionModern extends StatelessWidget {
  final String labourId;
  const _LabourReviewsSectionModern({required this.labourId});

  double _asDouble(dynamic v) =>
      (v is num) ? v.toDouble() : double.tryParse(v.toString()) ?? 0.0;

  @override
  Widget build(BuildContext context) {
    return _Section(
      title: "User Reviews",
      icon: Icons.reviews_rounded,
      child: StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
        stream: ReviewService.labourReviews(labourId),
        builder: (context, snap) {
          if (snap.hasError) {
            return _reviewInfo("Review stream error: ${snap.error}");
          }
          if (snap.connectionState == ConnectionState.waiting) {
            return const Center(
              child: Padding(
                padding: EdgeInsets.symmetric(vertical: 10),
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
            );
          }

          final docs = (snap.data?.docs ?? []).toList();

          docs.sort((a, b) {
            final at = a.data()["createdAt"];
            final bt = b.data()["createdAt"];
            final ad = (at is Timestamp) ? at.toDate() : DateTime(1970);
            final bd = (bt is Timestamp) ? bt.toDate() : DateTime(1970);
            return bd.compareTo(ad);
          });

          if (docs.isEmpty) {
            return _emptyReviews();
          }

          // Summary
          final ratings =
              docs.map((d) => _asDouble(d.data()["rating"])).toList();
          final avg = ratings.isEmpty
              ? 0.0
              : ratings.reduce((a, b) => a + b) / ratings.length;

          return Column(
            children: [
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.10),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: AppColors.border),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.star_rounded, color: Color(0xFFF5B400)),
                    const SizedBox(width: 8),
                    Text(
                      avg.toStringAsFixed(1),
                      style: const TextStyle(
                        fontWeight: FontWeight.w900,
                        color: AppColors.textDark,
                      ),
                    ),
                    const SizedBox(width: 10),
                    Text(
                      "${docs.length} reviews",
                      style: TextStyle(
                        fontWeight: FontWeight.w800,
                        color: AppColors.textDark.withOpacity(0.65),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 12),
              ...docs.map((d) => _reviewCard(d.data())).toList(),
            ],
          );
        },
      ),
    );
  }

  Widget _emptyReviews() {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: AppColors.primary.withOpacity(0.14),
              borderRadius: BorderRadius.circular(14),
            ),
            child: const Icon(Icons.chat_bubble_outline_rounded,
                color: AppColors.darkGreen),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              "No reviews yet. Be the first to hire and leave feedback.",
              style: TextStyle(
                fontWeight: FontWeight.w800,
                color: AppColors.textDark.withOpacity(0.70),
              ),
            ),
          ),
        ],
      ),
    );
  }

  static Widget _reviewInfo(String msg) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Text(
        msg,
        style: TextStyle(
          color: AppColors.textDark.withOpacity(0.65),
          fontWeight: FontWeight.w800,
        ),
      ),
    );
  }

  static Widget _reviewCard(Map<String, dynamic> m) {
    final by = (m["reviewByName"] ?? "User").toString();
    final sentiment = (m["sentiment"] ?? "").toString(); // good/bad
    final comment = (m["comment"] ?? "").toString();
    final ratingRaw = m["rating"];
    final rating = (ratingRaw is num) ? ratingRaw.toDouble() : 0.0;

    final createdAt = m["createdAt"];
    final dt = (createdAt is Timestamp) ? createdAt.toDate() : null;
    final dateTxt = (dt == null)
        ? ""
        : "${dt.year.toString().padLeft(4, "0")}-${dt.month.toString().padLeft(2, "0")}-${dt.day.toString().padLeft(2, "0")}";

    final isGood = sentiment == "good";

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
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
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  by,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
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
                  color: (isGood ? Colors.green : Colors.red).withOpacity(0.10),
                  borderRadius: BorderRadius.circular(999),
                  border: Border.all(
                    color:
                        (isGood ? Colors.green : Colors.red).withOpacity(0.25),
                  ),
                ),
                child: Text(
                  isGood ? "GOOD" : "BAD",
                  style: TextStyle(
                    color: isGood ? Colors.green : Colors.red,
                    fontWeight: FontWeight.w900,
                    fontSize: 11.5,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              const Icon(Icons.star_rounded,
                  color: Color(0xFFF5B400), size: 18),
              const SizedBox(width: 6),
              Text(
                rating.toStringAsFixed(1),
                style: const TextStyle(
                  fontWeight: FontWeight.w900,
                  color: AppColors.textDark,
                ),
              ),
              const SizedBox(width: 10),
              if (dateTxt.isNotEmpty)
                Text(
                  dateTxt,
                  style: TextStyle(
                    fontWeight: FontWeight.w800,
                    color: AppColors.textDark.withOpacity(0.55),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            comment.isEmpty ? "—" : comment,
            style: TextStyle(
              fontWeight: FontWeight.w800,
              color: AppColors.textDark.withOpacity(0.75),
              height: 1.3,
            ),
          ),
        ],
      ),
    );
  }
}

// ======================= SMALL UI PARTS =======================
class _Section extends StatelessWidget {
  final String title;
  final IconData icon;
  final Widget child;

  const _Section({
    required this.title,
    required this.icon,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return _CardShell(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 38,
                height: 38,
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.14),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: AppColors.border),
                ),
                child: Icon(icon, color: AppColors.darkGreen, size: 20),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    fontWeight: FontWeight.w900,
                    color: AppColors.textDark,
                    fontSize: 15,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          child,
        ],
      ),
    );
  }
}

class _CardShell extends StatelessWidget {
  final Widget child;
  const _CardShell({required this.child});

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

class _InfoTileModern extends StatelessWidget {
  final IconData icon;
  final String title;
  final String value;

  const _InfoTileModern({
    required this.icon,
    required this.title,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.fromLTRB(12, 12, 12, 12),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          Icon(icon, color: AppColors.darkGreen),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    fontWeight: FontWeight.w900,
                    fontSize: 11.5,
                    color: AppColors.textDark.withOpacity(0.6),
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  value,
                  style: const TextStyle(
                    fontWeight: FontWeight.w900,
                    fontSize: 14,
                    color: AppColors.textDark,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _StatPill extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;

  const _StatPill({
    required this.label,
    required this.value,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 12, 12, 12),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 16, color: AppColors.darkGreen),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  label,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontWeight: FontWeight.w900,
                    fontSize: 11.2,
                    color: AppColors.textDark.withOpacity(0.60),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            value,
            style: const TextStyle(
              fontWeight: FontWeight.w900,
              fontSize: 14.5,
              color: AppColors.textDark,
            ),
          ),
        ],
      ),
    );
  }
}

class _PillChip extends StatelessWidget {
  final IconData icon;
  final String text;
  const _PillChip({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.03),
            blurRadius: 12,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Row(
        children: [
          Icon(icon, size: 18, color: AppColors.darkGreen),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                fontWeight: FontWeight.w900,
                color: AppColors.textDark,
                fontSize: 12.5,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _TinyChip extends StatelessWidget {
  final String text;
  const _TinyChip({required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: AppColors.border),
      ),
      child: Text(
        text,
        style: TextStyle(
          fontWeight: FontWeight.w800,
          color: AppColors.textDark.withOpacity(0.75),
          fontSize: 12,
        ),
      ),
    );
  }
}

class _EmptyWorker extends StatelessWidget {
  final VoidCallback onBack;
  const _EmptyWorker({required this.onBack});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.fromLTRB(16, 14, 16, 18),
          decoration: const BoxDecoration(
            gradient: AppColors.heroGradient,
            borderRadius: BorderRadius.only(
              bottomLeft: Radius.circular(28),
              bottomRight: Radius.circular(28),
            ),
          ),
          child: Row(
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
              Text(
                "Worker Details",
                style: TextStyle(
                  color: Colors.white.withOpacity(0.95),
                  fontWeight: FontWeight.w900,
                  fontSize: 16,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 18),
        Expanded(
          child: Center(
            child: Text(
              "No worker data",
              style: TextStyle(
                color: AppColors.textDark.withOpacity(0.6),
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
        ),
      ],
    );
  }
}
