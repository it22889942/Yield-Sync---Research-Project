import 'package:flutter/material.dart';

import '../../services/review_service.dart';
import '../../utils/app_colors.dart';

/// Full-screen review & rating form (same data as [ReviewService.addReview]).
class AddReviewScreen extends StatefulWidget {
  final String bookingId;
  final String itemType; // "labour" | "equipment"
  final String itemId;
  final String itemName;
  final String ownerName;

  const AddReviewScreen({
    super.key,
    required this.bookingId,
    required this.itemType,
    required this.itemId,
    required this.itemName,
    required this.ownerName,
  });

  @override
  State<AddReviewScreen> createState() => _AddReviewScreenState();
}

class _AddReviewScreenState extends State<AddReviewScreen> {
  final _commentCtrl = TextEditingController();
  double _rating = 5.0;
  String _sentiment = "good";
  bool _submitting = false;

  @override
  void dispose() {
    _commentCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    final comment = _commentCtrl.text.trim();
    if (comment.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Please write a review.")),
      );
      return;
    }

    setState(() => _submitting = true);
    try {
      await ReviewService.addReview(
        bookingId: widget.bookingId,
        itemType: widget.itemType,
        itemId: widget.itemId,
        itemName: widget.itemName,
        ownerName: widget.ownerName,
        sentiment: _sentiment,
        rating: _rating,
        comment: comment,
      );
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Review saved ✅")),
      );
      Navigator.pop(context, true);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Save failed: $e")),
        );
      }
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.surface,
      appBar: AppBar(
        elevation: 0,
        backgroundColor: AppColors.surface,
        foregroundColor: AppColors.textDark,
        title: const Text(
          "Review & rating",
          style: TextStyle(fontWeight: FontWeight.w900),
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(18, 8, 18, 28),
        children: [
          Text(
            "Share feedback for this booking. Your rating helps others.",
            style: TextStyle(
              fontWeight: FontWeight.w700,
              color: AppColors.textDark.withOpacity(0.72),
              height: 1.35,
            ),
          ),
          const SizedBox(height: 16),
          Container(
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
              children: [
                _kv("Booking ID", widget.bookingId),
                _kv("Type", widget.itemType.toUpperCase()),
                _kv("ID", widget.itemId),
                _kv(
                  "Item",
                  widget.itemName.isEmpty ? "—" : widget.itemName,
                ),
                _kv(
                  widget.itemType == "labour" ? "Worker" : "Owner",
                  widget.ownerName.isEmpty ? "—" : widget.ownerName,
                ),
              ],
            ),
          ),
          const SizedBox(height: 18),
          Row(
            children: [
              Expanded(
                child: OutlinedButton(
                  onPressed: _submitting
                      ? null
                      : () => setState(() => _sentiment = "good"),
                  style: OutlinedButton.styleFrom(
                    side: BorderSide(
                      color: _sentiment == "good"
                          ? Colors.green
                          : AppColors.border,
                    ),
                    backgroundColor: _sentiment == "good"
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
                      color: _sentiment == "good"
                          ? Colors.green
                          : AppColors.textDark,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: OutlinedButton(
                  onPressed: _submitting
                      ? null
                      : () => setState(() => _sentiment = "bad"),
                  style: OutlinedButton.styleFrom(
                    side: BorderSide(
                      color:
                          _sentiment == "bad" ? Colors.red : AppColors.border,
                    ),
                    backgroundColor: _sentiment == "bad"
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
                      color: _sentiment == "bad"
                          ? Colors.red
                          : AppColors.textDark,
                    ),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              const Text(
                "Rating",
                style: TextStyle(
                  fontWeight: FontWeight.w900,
                  color: AppColors.textDark,
                ),
              ),
              const Spacer(),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(999),
                  border: Border.all(color: AppColors.border),
                ),
                child: Text(
                  _rating.toStringAsFixed(1),
                  style: const TextStyle(
                    fontWeight: FontWeight.w900,
                    color: AppColors.textDark,
                  ),
                ),
              ),
            ],
          ),
          Slider(
            value: _rating,
            min: 1,
            max: 5,
            divisions: 8,
            onChanged: _submitting
                ? null
                : (v) => setState(() => _rating = v),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _commentCtrl,
            maxLines: 4,
            enabled: !_submitting,
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
          const SizedBox(height: 22),
          SizedBox(
            width: double.infinity,
            height: 50,
            child: ElevatedButton.icon(
              onPressed: _submitting ? null : _submit,
              icon: _submitting
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.send_rounded),
              label: Text(
                _submitting ? "Saving…" : "Submit review",
                style: const TextStyle(fontWeight: FontWeight.w900),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primary,
                foregroundColor: AppColors.darkGreen,
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

  static Widget _kv(String k, String v) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 100,
            child: Text(
              k,
              style: TextStyle(
                fontWeight: FontWeight.w800,
                color: AppColors.textDark.withOpacity(0.60),
                fontSize: 12.5,
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
}
