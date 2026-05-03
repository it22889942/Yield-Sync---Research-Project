import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';
import '../../services/booking_service.dart';

class BookingRequestDialog extends StatefulWidget {
  final String labourId;
  const BookingRequestDialog({super.key, required this.labourId});

  @override
  State<BookingRequestDialog> createState() => _BookingRequestDialogState();
}

class _BookingRequestDialogState extends State<BookingRequestDialog> {
  DateTime _start = DateTime.now();
  DateTime _end = DateTime.now();
  bool _halfDay = false;
  bool _sending = false;
  String? _error;

  final _noteCtrl = TextEditingController();

  @override
  void dispose() {
    _noteCtrl.dispose();
    super.dispose();
  }

  String _fmt(DateTime d) {
    final y = d.year.toString().padLeft(4, "0");
    final m = d.month.toString().padLeft(2, "0");
    final day = d.day.toString().padLeft(2, "0");
    return "$y-$m-$day";
  }

  Future<void> _pickStart() async {
    FocusScope.of(context).unfocus(); // ✅ ADD THIS
    final d = await showDatePicker(
      context: context,
      initialDate: _start,
      firstDate: DateTime.now(),
      lastDate: DateTime.now().add(const Duration(days: 365)),
    );
    if (d == null) return;
    setState(() {
      _start = DateTime(d.year, d.month, d.day);
      if (_end.isBefore(_start)) _end = _start;
    });
  }

  Future<void> _pickEnd() async {
    FocusScope.of(context).unfocus(); // ✅ ADD THIS
    final d = await showDatePicker(
      context: context,
      initialDate: _end,
      firstDate: _start,
      lastDate: DateTime.now().add(const Duration(days: 365)),
    );
    if (d == null) return;
    setState(() => _end = DateTime(d.year, d.month, d.day));
  }

  Future<void> _send() async {
    setState(() {
      _sending = true;
      _error = null;
    });

    try {
      await BookingService.createRequest(
        labourId: widget.labourId,
        startDate: _start,
        endDate: _end,
        halfDay: _halfDay,
        note: _noteCtrl.text,
      );
      if (!mounted) return;
      Navigator.pop(context, true);
    } catch (e) {
      setState(() => _error = e.toString().replaceAll("Exception: ", ""));
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      insetPadding: const EdgeInsets.all(14),
      backgroundColor: const Color(0xFFEFEAF3),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 14),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Align(
              alignment: Alignment.centerLeft,
              child: Text(
                "Request Booking",
                style: TextStyle(
                  fontWeight: FontWeight.w900,
                  fontSize: 18,
                  color: AppColors.textDark,
                ),
              ),
            ),
            const SizedBox(height: 14),
            _dateRow("Start Date", _fmt(_start), _pickStart),
            const SizedBox(height: 10),
            _dateRow("End Date", _fmt(_end), _pickEnd),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: RadioListTile<bool>(
                    value: false,
                    groupValue: _halfDay,
                    onChanged:
                        _sending ? null : (v) => setState(() => _halfDay = v!),
                    title: const Text("Day",
                        style: TextStyle(fontWeight: FontWeight.w800)),
                  ),
                ),
                Expanded(
                  child: RadioListTile<bool>(
                    value: true,
                    groupValue: _halfDay,
                    onChanged:
                        _sending ? null : (v) => setState(() => _halfDay = v!),
                    title: const Text("Half-day",
                        style: TextStyle(fontWeight: FontWeight.w800)),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _noteCtrl,
              maxLines: 3,
              decoration: InputDecoration(
                hintText: "Note (optional)",
                filled: true,
                fillColor: Colors.white,
                contentPadding: const EdgeInsets.all(14),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(14),
                  borderSide: BorderSide.none,
                ),
              ),
            ),
            if (_error != null) ...[
              const SizedBox(height: 10),
              Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  _error!,
                  style: const TextStyle(
                    color: Colors.red,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
            ],
            const SizedBox(height: 14),
            Row(
              children: [
                TextButton(
                  onPressed:
                      _sending ? null : () => Navigator.pop(context, false),
                  child: const Text("Cancel",
                      style: TextStyle(fontWeight: FontWeight.w900)),
                ),
                const Spacer(),
                ElevatedButton(
                  onPressed: _sending ? null : _send,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    foregroundColor: AppColors.darkGreen,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(999),
                    ),
                  ),
                  child: Text(
                    _sending ? "Sending..." : "Send Request",
                    style: const TextStyle(fontWeight: FontWeight.w900),
                  ),
                ),
              ],
            )
          ],
        ),
      ),
    );
  }

  Widget _dateRow(String title, String value, VoidCallback onTap) {
    return InkWell(
      onTap: _sending ? null : onTap,
      borderRadius: BorderRadius.circular(14),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
        decoration: BoxDecoration(
          color: const Color(0xFFEFEAF3),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: Colors.black12),
        ),
        child: Row(
          children: [
            Expanded(
              child: Text(
                title,
                style: const TextStyle(
                  fontWeight: FontWeight.w800,
                  color: AppColors.textDark,
                ),
              ),
            ),
            Text(
              value,
              style: TextStyle(
                fontWeight: FontWeight.w900,
                color: AppColors.textDark.withOpacity(0.7),
              ),
            ),
            const SizedBox(width: 10),
            const Icon(Icons.calendar_month_rounded),
          ],
        ),
      ),
    );
  }
}
