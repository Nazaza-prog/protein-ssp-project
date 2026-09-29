let eventType = 0;
let basePrice = 0;

while (Number.isNaN(eventType) || eventType < 1 || eventType > 3) {
    eventType = +prompt(
        "Оберіть тип події:\n" +
        "1 - Кіно 150 грн\n" +
        "2 - Театр 220 грн\n" +
        "3 - Концерт 350 грн"
    );
    switch (eventType) {
        case 1:
            basePrice = 150;
            break;
        case 2:
            basePrice = 220;
            break;
        case 3:
            basePrice = 350;
            break;
        default:
            alert("Не то");
            continue;
    }
    break;
}

let dayType = 0;
while (dayType !== 1 && dayType !== 2) {
    dayType = +prompt(
        "Оберіть день:\n" +
        "1 — Будній\n" +
        "2 — Вихідний (15% знижки)"
    );
    if (dayType !== 1 && dayType !== 2) {
        alert("Не то. Виберіть 1-будній або 2-вихідний.");
    }
}
let ticketBasePrice = basePrice;
if (dayType === 2) {
    ticketBasePrice = basePrice * 1.15;
}

let ticketsCount = 0;
while (Number.isNaN(ticketsCount) || ticketsCount < 1 || ticketsCount > 6) {
    ticketsCount = +prompt("Введіть кількість квитків від 1 до 6 ");
    if (Number.isNaN(ticketsCount) ||ticketsCount < 1 || ticketsCount > 6) {
        alert("Не то");
    }
}

let processedTickets = 0;
let freeTicketsCount = 0;
let discountedTicketsCount = 0;
let fullPriceTicketsCount = 0;
let totalSum = 0;

for (let i = 1; i <= ticketsCount; i++) {
    let input = prompt(`Квиток #${i}. Введіть вік відвідувача:`);
    if (input === null) {
        alert("Оформлення перервано.");
        break;
    }
    let age = +input;


    while (Number.isNaN(age) || age < 0 || age > 120) {
        alert("Не то");
        input = prompt(`Квиток ${i}. Введіть вік відвідувача:`);

        if (input === null) break;
        age = +input;
    }

    processedTickets++;

    let discountPercent = 0;

    if (age >= 0 && age <= 5) {
        freeTicketsCount++;
        console.log(`Квиток #${i}: Безкоштовно (вік: ${age})`);
        continue;

    } else if (age >= 6 && age <= 12) {
        discountPercent = 50;
        discountedTicketsCount++;

    } else if (age >= 13 && age <= 17) {
        discountPercent = 20;
        discountedTicketsCount++;

    } else if (age >= 18 && age <= 59) {
        if (age <= 25) {
            let hasStudentCard = confirm(`Квиток #${i}: Чи є у вас студентський квиток?`);
            if (hasStudentCard) {
                discountPercent = 10;
                discountedTicketsCount++;
            } else {
                fullPriceTicketsCount++;
            }
        } else {
            fullPriceTicketsCount++;
        }
    } else if (age >= 60) {
        discountPercent = 25;
        discountedTicketsCount++;
    }

    let currentTicketPrice = ticketBasePrice * (1 - discountPercent / 100);
    totalSum += currentTicketPrice;

    console.log(`Квиток #${i}: ${currentTicketPrice.toFixed(2)} грн (знижка ${discountPercent}%)`);
}
let finalSum = totalSum;
if (finalSum > 1000) {
    finalSum = finalSum * 0.95;
    console.log("Додаткова знижка 5%.");
}



console.log("сервіс TicketFlow");
console.log(`Оброблено квитків: ${processedTickets}`);
console.log(`Безкоштовних: ${freeTicketsCount}`);
console.log(`Зі знижкою: ${discountedTicketsCount}`);
console.log(`За повною ціною: ${fullPriceTicketsCount}`);

console.log(`Загальна сума до сплати: ${finalSum} грн`);


alert(
    `сервіс TicketFlow:\n` +
    `Оброблено квитків: ${processedTickets}\n` +
    `Безкоштовних: ${freeTicketsCount}\n` +
    `Зі знижкою: ${discountedTicketsCount}\n` +
    `Повна ціна: ${fullPriceTicketsCount}\n` +
    `Загальна вартість: ${finalSum} грн`
);