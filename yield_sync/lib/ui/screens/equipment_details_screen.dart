import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';

import '../../utils/app_colors.dart';
import '../../utils/equipment_image_asset.dart';
import '../widgets/app_shell.dart';
import '../../services/equipment_api.dart';
import '../../services/equipment_booking_service.dart';
import '../../services/review_service.dart';

class EquipmentDetailsScreen extends StatefulWidget {
  const EquipmentDetailsScreen({super.key});

  @override
  State<EquipmentDetailsScreen> createState() => _EquipmentDetailsScreenState();
}

class _EquipmentDetailsScreenState extends State<EquipmentDetailsScreen> {
  String? _id;
  bool _loading = true;
  String? _error;
  EquipmentDetails? _details;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final arg = ModalRoute.of(context)?.settings.arguments;
    if (arg is String) _id = arg;
    _fetch();
  }

  Future<void> _fetch() async {
    final id = _id;
    if (id == null || id.isEmpty) {
      setState(() {
        _loading = false;
        _error = "Equipment ID not found";
      });
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final d = await EquipmentApi.getItem(id);
      if (!mounted) return;
      setState(() {
        _details = d;
        _loading = false;
        _error = null;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  // ================= BOOKING (unchanged) =================
  Future<void> _showBookingDialog(EquipmentDetails e) async {
    DateTime? startDate;
    DateTime? endDate;
    String type = "full_day";
    String note = "";
    bool sending = false;

    await showDialog(
      context: context,
      barrierDismissible: !sending,
      builder: (_) {
        return StatefulBuilder(
          builder: (context, setStateDialog) {
            Future<void> pickStart() async {
              final d = await showDatePicker(
                context: context,
                initialDate: DateTime.now(),
                firstDate: DateTime.now(),
                lastDate: DateTime(2030),
              );
              if (d != null) setStateDialog(() => startDate = d);
            }

            Future<void> pickEnd() async {
              final d = await showDatePicker(
                context: context,
                initialDate: startDate ?? DateTime.now(),
                firstDate: startDate ?? DateTime.now(),
                lastDate: DateTime(2030),
              );
              if (d != null) setStateDialog(() => endDate = d);
            }

            Future<void> submit() async {
              if (startDate == null || endDate == null) {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text("Please select dates")),
                );
                return;
              }

              setStateDialog(() => sending = true);

              try {
                await EquipmentBookingService.createBooking(
                  equipmentId: e.id,
                  startDate: startDate!,
                  endDate: endDate!,
                  type: type,
                  note: note,
                );

                if (context.mounted) Navigator.pop(context);

                if (mounted) {
                  ScaffoldMessenger.of(this.context).showSnackBar(
                    const SnackBar(content: Text("Booking request sent ✅")),
                  );
                }
              } catch (err) {
                if (mounted) {
                  ScaffoldMessenger.of(this.context).showSnackBar(
                    SnackBar(content: Text("Booking failed: $err")),
                  );
                }
              } finally {
                if (context.mounted) setStateDialog(() => sending = false);
              }
            }

            return AlertDialog(
              title: const Text("Book Equipment"),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  ListTile(
                    title: Text(
                      startDate == null
                          ? "Select Start Date"
                          : startDate.toString().split(" ")[0],
                    ),
                    trailing: const Icon(Icons.calendar_today),
                    onTap: sending ? null : pickStart,
                  ),
                  ListTile(
                    title: Text(
                      endDate == null
                          ? "Select End Date"
                          : endDate.toString().split(" ")[0],
                    ),
                    trailing: const Icon(Icons.calendar_today),
                    onTap: sending ? null : pickEnd,
                  ),
                  const SizedBox(height: 10),
                  DropdownButtonFormField<String>(
                    value: type,
                    decoration:
                        const InputDecoration(labelText: "Booking Type"),
                    items: const [
                      DropdownMenuItem(
                          value: "full_day", child: Text("Full Day")),
                      DropdownMenuItem(
                          value: "half_day", child: Text("Half Day")),
                      DropdownMenuItem(
                          value: "per_hour", child: Text("Per Hour")),
                    ],
                    onChanged: sending
                        ? null
                        : (v) => setStateDialog(() => type = v ?? "full_day"),
                  ),
                  const SizedBox(height: 10),
                  TextField(
                    enabled: !sending,
                    decoration: const InputDecoration(
                      labelText: "Note (optional)",
                    ),
                    onChanged: (v) => note = v,
                  ),
                ],
              ),
              actions: [
                TextButton(
                  onPressed: sending ? null : () => Navigator.pop(context),
                  child: const Text("Cancel"),
                ),
                ElevatedButton(
                  onPressed: sending ? null : submit,
                  child: Text(sending ? "Booking..." : "Book"),
                ),
              ],
            );
          },
        );
      },
    );
  }

  // ================= image helpers (NEW) =================

  IconData _iconForType(String t) {
    final s = t.toLowerCase();
    if (s.contains("tractor")) return Icons.agriculture_rounded;
    if (s.contains("harvest") || s.contains("combine"))
      return Icons.grass_rounded;
    if (s.contains("pump") || s.contains("water")) return Icons.water_rounded;
    if (s.contains("spray")) return Icons.water_drop_rounded;
    if (s.contains("plough")) return Icons.handyman_rounded;
    if (s.contains("seed")) return Icons.spa_rounded;
    if (s.contains("trailer")) return Icons.local_shipping_rounded;
    if (s.contains("transplant")) return Icons.eco_rounded;
    if (s.contains("fertilizer") || s.contains("spreader")) {
      return Icons.scatter_plot_rounded;
    }
    if (s.contains("grass") || s.contains("cutter"))
      return Icons.content_cut_rounded;
    return Icons.build_rounded;
  }

  Widget _heroFallback(String type) {
    return Container(
      color: AppColors.surface,
      child: Center(
        child: Icon(
          _iconForType(type),
          size: 84,
          color: AppColors.darkGreen,
        ),
      ),
    );
  }

  // ================= UI (modern) =================
  @override
  Widget build(BuildContext context) {
    return AppShell(
      currentIndex: 0,
      child: Column(
        children: [
          _modernHeader(),
          Expanded(
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : _error != null
                    ? _errorState(_error!)
                    : _modernDetailsView(_details!),
          ),
        ],
      ),
    );
  }

  Widget _modernHeader() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
      decoration: const BoxDecoration(
        gradient: AppColors.heroGradient,
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(28),
          bottomRight: Radius.circular(28),
        ),
      ),
      child: Row(
        children: [
          _roundIconBtn(
            icon: Icons.arrow_back_ios_new_rounded,
            onTap: () => Navigator.pop(context),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              "Equipment Details",
              style: TextStyle(
                color: Colors.white.withOpacity(0.95),
                fontWeight: FontWeight.w900,
                fontSize: 16,
              ),
            ),
          ),
          _roundIconBtn(
            icon: Icons.refresh_rounded,
            onTap: _fetch,
          ),
        ],
      ),
    );
  }

  Widget _modernDetailsView(EquipmentDetails e) {
    final type = (e.equipmentType.isEmpty) ? "Equipment" : e.equipmentType;
    final heroAsset = EquipmentImageAsset.resolve(
      equipmentType: e.equipmentType,
      forCrop: e.forCrop,
      id: e.id,
    );

    return Stack(
      children: [
        ListView(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 120),
          children: [
            // HERO CARD
            Container(
              height: 220,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(22),
                border: Border.all(color: AppColors.border),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.08),
                    blurRadius: 26,
                    offset: const Offset(0, 16),
                  ),
                ],
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(22),
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Image.asset(
                      heroAsset,
                      fit: BoxFit.cover,
                      errorBuilder: (_, __, ___) => _heroFallback(type),
                    ),

                    Container(
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                          colors: [
                            Colors.black.withOpacity(0.12),
                            Colors.black.withOpacity(0.55),
                          ],
                        ),
                      ),
                    ),

                    Positioned(
                      left: 14,
                      right: 14,
                      bottom: 14,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // readable badge on image
                          _badge(
                            icon: Icons.category_rounded,
                            text: type,
                          ),
                          const SizedBox(height: 10),
                          Text(
                            type,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w900,
                              fontSize: 22,
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            "Equipment ID: ${e.id}",
                            style: TextStyle(
                              color: Colors.white.withOpacity(0.78),
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 14),

            // PRICE CARDS
            Row(
              children: [
                Expanded(
                  child: _statCard(
                    title: "Daily Rate",
                    value: "LKR ${e.dailyRate}",
                    icon: Icons.payments_rounded,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _statCard(
                    title: "Hourly Rate",
                    value: "LKR ${e.hourlyRate}/hr",
                    icon: Icons.timer_rounded,
                  ),
                ),
              ],
            ),

            const SizedBox(height: 12),

            // OWNER CARD
            _whiteCard(
              child: Row(
                children: [
                  Container(
                    width: 46,
                    height: 46,
                    decoration: BoxDecoration(
                      color: AppColors.primary.withOpacity(0.16),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: const Icon(Icons.person_rounded,
                        color: AppColors.darkGreen),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      e.ownerName,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontWeight: FontWeight.w900,
                        color: AppColors.textDark,
                        fontSize: 15.5,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 12),

            // DETAILS (nice info rows)
            _whiteCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    "Overview",
                    style: TextStyle(
                      fontWeight: FontWeight.w900,
                      fontSize: 15.5,
                      color: AppColors.textDark,
                    ),
                  ),
                  const SizedBox(height: 10),
                  _infoRow("Type", type),
                  const SizedBox(height: 8),
                  _infoRow("Equipment ID", e.id),
                ],
              ),
            ),

            // REVIEWS
            const SizedBox(height: 16),
            _reviewsSectionEquipment(e.id),
          ],
        ),

        // STICKY BOOK CTA
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
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          "LKR ${e.dailyRate}/day",
                          style: const TextStyle(
                            fontWeight: FontWeight.w900,
                            color: AppColors.textDark,
                            fontSize: 14.5,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          "Tap to request booking",
                          style: TextStyle(
                            fontWeight: FontWeight.w700,
                            color: AppColors.textDark.withOpacity(0.55),
                            fontSize: 12.2,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: SizedBox(
                      height: 50,
                      child: ElevatedButton.icon(
                        onPressed: () => _showBookingDialog(e),
                        icon: const Icon(Icons.calendar_month_rounded),
                        label: const Text(
                          "Book Now",
                          style: TextStyle(fontWeight: FontWeight.w900),
                        ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppColors.primary,
                          foregroundColor: AppColors.darkGreen,
                          elevation: 0,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16),
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        )
      ],
    );
  }

  // ================= REVIEWS (your same logic, UI improved) =================
  Widget _reviewsSectionEquipment(String equipmentId) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          "User Reviews",
          style: TextStyle(
            fontWeight: FontWeight.w900,
            fontSize: 16,
            color: AppColors.textDark,
          ),
        ),
        const SizedBox(height: 10),
        StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
          stream: ReviewService.equipmentReviews(equipmentId),
          builder: (context, snap) {
            if (snap.hasError) {
              return _reviewInfo("Review stream error: ${snap.error}");
            }
            if (snap.connectionState == ConnectionState.waiting) {
              return const Center(
                child: Padding(
                  padding: EdgeInsets.all(12),
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
              return _reviewInfo("No reviews yet");
            }

            return Column(
              children: docs.map((d) {
                final m = d.data();
                return _reviewCard(m);
              }).toList(),
            );
          },
        ),
      ],
    );
  }

  // ================= small UI helpers =================
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

  // ✅ improved readable badge on image
  Widget _badge({required IconData icon, required String text}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.92),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withOpacity(0.95)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.10),
            blurRadius: 10,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: AppColors.darkGreen),
          const SizedBox(width: 8),
          ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 220),
            child: Text(
              text,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                color: AppColors.darkGreen,
                fontWeight: FontWeight.w900,
                fontSize: 12,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _statCard(
      {required String title, required String value, required IconData icon}) {
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
      child: Row(
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: AppColors.primary.withOpacity(0.16),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Icon(icon, color: AppColors.darkGreen),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    fontWeight: FontWeight.w800,
                    color: AppColors.textDark.withOpacity(0.60),
                    fontSize: 12,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  value,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontWeight: FontWeight.w900,
                    color: AppColors.textDark,
                    fontSize: 16,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _whiteCard({required Widget child}) {
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

  Widget _infoRow(String k, String v) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
      decoration: BoxDecoration(
        color: AppColors.primary.withOpacity(0.10),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              k,
              style: TextStyle(
                color: AppColors.textDark.withOpacity(0.70),
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
          Flexible(
            child: Text(
              v,
              textAlign: TextAlign.right,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                color: AppColors.darkGreen,
                fontWeight: FontWeight.w900,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _errorState(String msg) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: AppColors.border),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error_outline_rounded,
                  color: Colors.red, size: 34),
              const SizedBox(height: 10),
              Text(
                msg,
                style: const TextStyle(
                  fontWeight: FontWeight.w800,
                  color: AppColors.textDark,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 10),
              SizedBox(
                height: 46,
                child: ElevatedButton.icon(
                  onPressed: _fetch,
                  icon: const Icon(Icons.refresh_rounded),
                  label: const Text(
                    "Retry",
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
              )
            ],
          ),
        ),
      ),
    );
  }

  Widget _reviewInfo(String msg) {
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
          color: AppColors.textDark.withOpacity(0.65),
          fontWeight: FontWeight.w800,
        ),
      ),
    );
  }

  Widget _reviewCard(Map<String, dynamic> m) {
    final by = (m["reviewByName"] ?? "User").toString();
    final sentiment = (m["sentiment"] ?? "").toString();
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
              _stars(rating),
              const SizedBox(width: 8),
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
            ),
          ),
        ],
      ),
    );
  }

  Widget _stars(double rating) {
    final r = rating.clamp(0, 5);
    final full = r.floor();
    final half = (r - full) >= 0.5;

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: List.generate(5, (i) {
        if (i < full) {
          return const Icon(Icons.star_rounded,
              size: 18, color: Color(0xFFF5B400));
        }
        if (i == full && half) {
          return const Icon(Icons.star_half_rounded,
              size: 18, color: Color(0xFFF5B400));
        }
        return Icon(Icons.star_border_rounded,
            size: 18, color: Colors.black.withOpacity(0.25));
      }),
    );
  }
}
