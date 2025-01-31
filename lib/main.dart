import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'XRP Predictor',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: PredictorScreen(),
    );
  }
}

class PredictorScreen extends StatefulWidget {
  @override
  _PredictorScreenState createState() => _PredictorScreenState();
}

class _PredictorScreenState extends State<PredictorScreen> {
  final TextEditingController openController = TextEditingController();
  final TextEditingController closeController = TextEditingController();
  String? predictedPrice;

  Future<void> getPrediction() async {
    final String apiUrl = 'http://127.0.0.1:5000/predict'; // Změň IP na svou API adresu
    final double open = double.tryParse(openController.text) ?? 0.0;
    final double close = double.tryParse(closeController.text) ?? 0.0;

    try {
      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'open': open, 'close': close}),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          predictedPrice = data['predicted_close'].toString();
        });
      } else {
        setState(() {
          predictedPrice = 'Error: ${response.reasonPhrase}';
        });
      }
    } catch (e) {
      setState(() {
        predictedPrice = 'Error: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('XRP Price Predictor'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: openController,
              decoration: InputDecoration(labelText: 'Open Price'),
              keyboardType: TextInputType.number,
            ),
            TextField(
              controller: closeController,
              decoration: InputDecoration(labelText: 'Close Price'),
              keyboardType: TextInputType.number,
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: getPrediction,
              child: Text('Predict Price'),
            ),
            SizedBox(height: 20),
            if (predictedPrice != null)
              Text(
                'Predicted Close Price: $predictedPrice',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
          ],
        ),
      ),
    );
  }
}
