using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;

namespace MyApp.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class PaymentController : ControllerBase
    {
        private readonly IPaymentService _paymentService;
        private readonly ILogger<PaymentController> _logger;

        public PaymentController(IPaymentService paymentService, ILogger<PaymentController> logger)
        {
            _paymentService = paymentService;
            _logger = logger;
        }

        [HttpPost]
        [Route("charge")]
        public async Task<IActionResult> Charge(ChargeRequest request)
        {
            var result = await _paymentService.ChargeAsync(request);
            _logger.LogInformation("Charged: {Id}", result.Id);
            return Ok(result);
        }

        [HttpGet("{id}")]
        public async Task<IActionResult> GetStatus(string id)
        {
            var status = await _paymentService.GetStatusAsync(id);
            if (status == null)
                return NotFound();
            return Ok(status);
        }

        private bool ValidateRequest(ChargeRequest req) => req != null && req.Amount > 0;
    }

    public interface IPaymentService
    {
        Task<ChargeResult> ChargeAsync(ChargeRequest request);
        Task<PaymentStatus> GetStatusAsync(string id);
    }
}
