import 'package:flutter/material.dart';
import '../../services/market_api.dart';
import '../../utils/app_colors.dart';
import '../widgets/app_shell.dart';

class MarketUpdateDataScreen extends StatefulWidget {
  const MarketUpdateDataScreen({super.key});

  @override
  State<MarketUpdateDataScreen> createState() => _MarketUpdateDataScreenState();
}

class _MarketUpdateDataScreenState extends State<MarketUpdateDataScreen> {
  bool _loading = true;
  bool _updating = false;
  bool _retraining = false;
  String _statusText = "";
  String? _error;
  String _lastEntryDate = "-";
  String _nextUpdateDate = "-";

  @override
  void initState() {
    super.initState();
    _loadStatus();
  }

  Future<void> _loadStatus() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final res = await MarketApi.trends(maxPoints: 1, recent: 1);
      final stats = (res["stats"] as Map?)?.cast<String, dynamic>() ?? {};
      final lastDateRaw = (stats["last_date"] ?? "").toString();

      final lastDate = DateTime.tryParse(lastDateRaw);
      final nextDate = lastDate?.add(const Duration(days: 7));
      final now = DateTime.now();

      setState(() {
        _lastEntryDate = _formatDate(lastDate);
        _nextUpdateDate = _formatDate(nextDate);
        if (lastDate == null) {
          _statusText = "Could not detect the latest entry date.";
        } else if (now.difference(lastDate).inDays <= 7) {
          _statusText = "Data is up to date.";
        } else {
          _statusText = "New weekly data may be available.";
        }
      });
    } catch (e) {
      setState(() => _error = "$e");
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _fetchLatestWeek() async {
    setState(() {
      _updating = true;
      _error = null;
    });

    try {
      final result = await MarketApi.updateSmartWeek();
      if (!mounted) return;
      final message = (result["message"] ?? "Week update completed.").toString();
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(message)),
      );
      await _loadStatus();
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = "$e");
    } finally {
      if (mounted) setState(() => _updating = false);
    }
  }

  Future<void> _retrainModels() async {
    setState(() {
      _retraining = true;
      _error = null;
    });

    try {
      final result = await MarketApi.retrainModels();
      if (!mounted) return;
      final modelsTrained = result["models_trained"]?.toString() ?? "N/A";
      final duration = result["duration_minutes"]?.toString() ?? "N/A";
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content:
              Text("Retrain done. Models: $modelsTrained, minutes: $duration"),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = "$e");
    } finally {
      if (mounted) setState(() => _retraining = false);
    }
  }

  String _formatDate(DateTime? date) {
    if (date == null) return "-";
    final y = date.year.toString().padLeft(4, '0');
    final m = date.month.toString().padLeft(2, '0');
    final d = date.day.toString().padLeft(2, '0');
    return "$y-$m-$d";
  }

  @override
  Widget build(BuildContext context) {
    return AppShell(
      currentIndex: 0,
      child: Column(
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
            child: Column(
              children: [
                Row(
                  children: [
                    IconButton(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(Icons.arrow_back_ios_new_rounded),
                      color: Colors.white,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      "Update Data",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.92),
                        fontWeight: FontWeight.w900,
                        fontSize: 16,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                Text(
                  "Sync Weekly Data",
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.98),
                    fontWeight: FontWeight.w900,
                    fontSize: 22,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          Expanded(
            child: ListView(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              children: [
                _actionButton(
                  icon: Icons.sync_rounded,
                  text: "Fetch Latest Week",
                  busy: _updating,
                  onTap: _updating || _loading ? null : _fetchLatestWeek,
                ),
                const SizedBox(height: 12),
                _actionButton(
                  icon: Icons.model_training_rounded,
                  text: "Retrain Models",
                  busy: _retraining,
                  onTap: _retraining || _loading ? null : _retrainModels,
                ),
                const SizedBox(height: 12),
                if (_loading)
                  const Center(
                    child: Padding(
                      padding: EdgeInsets.all(16),
                      child: CircularProgressIndicator(),
                    ),
                  )
                else ...[
                  _infoCard(
                    icon: Icons.check_box_rounded,
                    text: "$_statusText Last entry: $_lastEntryDate",
                  ),
                  const SizedBox(height: 12),
                  _infoCard(
                    icon: Icons.calendar_month_rounded,
                    text: "Next update available after: $_nextUpdateDate",
                  ),
                ],
                if (_error != null) ...[
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.red.withOpacity(0.08),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: Colors.red.withOpacity(0.2)),
                    ),
                    child: Text(
                      _error!,
                      style: const TextStyle(
                        color: Colors.red,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _actionButton({
    required IconData icon,
    required String text,
    required bool busy,
    required VoidCallback? onTap,
  }) {
    return SizedBox(
      height: 52,
      child: ElevatedButton.icon(
        onPressed: onTap,
        icon: busy
            ? const SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  color: Colors.white,
                ),
              )
            : Icon(icon),
        label: Text(
          text,
          style: const TextStyle(fontWeight: FontWeight.w900),
        ),
        style: ElevatedButton.styleFrom(
          backgroundColor: AppColors.darkGreen,
          foregroundColor: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
        ),
      ),
    );
  }

  Widget _infoCard({required IconData icon, required String text}) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.primary.withOpacity(0.14),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: AppColors.darkGreen),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(
                fontWeight: FontWeight.w800,
                color: AppColors.textDark,
                height: 1.35,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
