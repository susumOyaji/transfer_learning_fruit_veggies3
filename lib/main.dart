import 'package:flutter/material.dart';
import 'home.dart';
import 'Neon.dart';

void main() {
  runApp(const MyApp1());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fruit Recognition',
      home: Home(),
      debugShowCheckedModeBanner: false,
    );
  }
}
